from dataclasses import dataclass, field
import contextlib
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from groot.vla.model.dreamzero.action_head.wan_amd_distill import (
    WANPolicyHeadAMD,
    WANPolicyHeadAMDConfig,
    _HiddenStateCapture,
    _get_dit_blocks,
    _lora_disabled,
)
from einops import rearrange
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


@dataclass
class WANPolicyHeadAMDTrustRegionConfig(WANPolicyHeadAMDConfig):
    enable_token_weighting: bool = field(
        default=True,
        metadata={"help": "If False, falls back to uniform-weighted AMD distillation."},
    )
    weight_predictor_hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden width of the per-token weight MLP."},
    )
    weight_predictor_num_layers: int = field(
        default=2,
        metadata={"help": "Number of hidden layers in the per-token weight MLP."},
    )
    weight_predictor_use_teacher_hidden: bool = field(
        default=False,
        metadata={
            "help": "Feed the teacher's intermediate noisy-video hidden state into the "
            "weight predictor as an additional content cue."
        },
    )
    motion_prior_strength: float = field(
        default=4.0,
        metadata={
            "help": "Initial multiplier alpha on the motion prior. Higher = more aggressive "
            "down-weighting of distillation in moving regions."
        },
    )
    motion_prior_learnable: bool = field(
        default=True,
        metadata={"help": "Whether the motion-prior multiplier alpha is a learnable scalar."},
    )
    initial_weight_target: float = field(
        default=0.7,
        metadata={
            "help": "Target mean weight at initialisation. Bias of the predictor is set so "
            "that for zero motion and zero MLP adjustment, w == initial_weight_target."
        },
    )
    weight_mean_target: float = field(
        default=0.5,
        metadata={
            "help": "Desired long-run average distillation weight. The reg loss penalises "
            "drift away from this target."
        },
    )
    weight_mean_reg_coeff: float = field(
        default=0.05,
        metadata={
            "help": "Coefficient on the mean-weight regulariser. Prevents the predictor "
            "from collapsing toward zero everywhere."
        },
    )
    weight_entropy_coeff: float = field(
        default=0.0,
        metadata={
            "help": "Coefficient on a Bernoulli-entropy regulariser per token. Pushes "
            "weights toward uncertain (entropy-high) values to discourage trivial 0/1."
        },
    )
    weight_min: float = field(
        default=0.0,
        metadata={"help": "Minimum allowed weight after sigmoid (clamp lower bound)."},
    )
    weight_max: float = field(
        default=1.0,
        metadata={"help": "Maximum allowed weight after sigmoid (clamp upper bound)."},
    )


def _build_mlp(in_dim: int, hidden_dim: int, num_layers: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(max(1, num_layers)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TokenWeightPredictor(nn.Module):
    def __init__(
        self,
        teacher_hidden_dim: int = 0,
        use_teacher_hidden: bool = False,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_weight_target: float = 0.7,
        motion_prior_strength: float = 4.0,
        motion_prior_learnable: bool = True,
        weight_min: float = 0.0,
        weight_max: float = 1.0,
    ):
        super().__init__()

        if not (0.0 < init_weight_target < 1.0):
            raise ValueError(
                f"init_weight_target must be in (0, 1), got {init_weight_target}"
            )
        if weight_min < 0.0 or weight_max > 1.0 or weight_min >= weight_max:
            raise ValueError(
                f"Invalid (weight_min={weight_min}, weight_max={weight_max})."
            )

        self.use_teacher_hidden = use_teacher_hidden
        self.teacher_hidden_dim = teacher_hidden_dim if use_teacher_hidden else 0
        self.weight_min = weight_min
        self.weight_max = weight_max

        in_dim = 1 + self.teacher_hidden_dim
        self.mlp = _build_mlp(in_dim, hidden_dim, num_layers, 1)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        alpha_init = torch.tensor(float(motion_prior_strength))
        if motion_prior_learnable:
            self.alpha = nn.Parameter(alpha_init)
        else:
            self.register_buffer("alpha", alpha_init)

        bias_init = math.log(init_weight_target / (1.0 - init_weight_target))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(
        self,
        motion: torch.Tensor,
        teacher_hidden_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = motion.unsqueeze(-1)
        if self.use_teacher_hidden:
            if teacher_hidden_grid is None:
                raise ValueError(
                    "use_teacher_hidden=True but teacher_hidden_grid is None."
                )
            if teacher_hidden_grid.shape[-1] != self.teacher_hidden_dim:
                raise ValueError(
                    "teacher_hidden_grid last dim mismatch: "
                    f"got {teacher_hidden_grid.shape[-1]}, expected {self.teacher_hidden_dim}."
                )
            feats = torch.cat(
                [feats, teacher_hidden_grid.to(dtype=feats.dtype)], dim=-1
            )

        adj = self.mlp(feats).squeeze(-1)
        logits = self.bias - self.alpha * motion + adj
        w = torch.sigmoid(logits)
        if self.weight_min > 0.0 or self.weight_max < 1.0:
            w = self.weight_min + (self.weight_max - self.weight_min) * w
        return w


class WANPolicyHeadAMDTrustRegion(WANPolicyHeadAMD):
    config_class = WANPolicyHeadAMDTrustRegionConfig

    def __init__(self, config: WANPolicyHeadAMDTrustRegionConfig):
        super().__init__(config)
        self.tr_config: WANPolicyHeadAMDTrustRegionConfig = config

        self.weight_predictor = TokenWeightPredictor(
            teacher_hidden_dim=self.model.dim if hasattr(self.model, "dim") else 0,
            use_teacher_hidden=config.weight_predictor_use_teacher_hidden,
            hidden_dim=config.weight_predictor_hidden_dim,
            num_layers=config.weight_predictor_num_layers,
            init_weight_target=config.initial_weight_target,
            motion_prior_strength=config.motion_prior_strength,
            motion_prior_learnable=config.motion_prior_learnable,
            weight_min=config.weight_min,
            weight_max=config.weight_max,
        )
        for p in self.weight_predictor.parameters():
            p.requires_grad = True

        self._tr_logged = False

    def _select_block(self) -> nn.Module:
        blocks = _get_dit_blocks(self.model)
        idx = self.tr_config.distill_hidden_layer
        if idx < 0:
            idx = len(blocks) + idx
        idx = max(0, min(len(blocks) - 1, idx))
        return blocks[idx]

    @staticmethod
    def _compute_motion_prior(latents_btchw: torch.Tensor) -> torch.Tensor:
        if latents_btchw.shape[1] <= 1:
            B, F_, _, H, W = latents_btchw.shape
            return torch.zeros(B, F_, H, W, device=latents_btchw.device, dtype=latents_btchw.dtype)

        diff = (latents_btchw[:, 1:] - latents_btchw[:, :-1]).abs().mean(dim=2)
        zeros = torch.zeros_like(diff[:, :1])
        motion = torch.cat([zeros, diff], dim=1)

        max_per_sample = motion.flatten(1).max(dim=1, keepdim=True)[0].clamp_min(1e-6)
        motion = motion / max_per_sample[..., None, None]
        return motion.detach()

    @staticmethod
    def _hidden_to_grid(
        hidden: torch.Tensor,
        seq_len: int,
        F_lat: int,
        H_lat: int,
        W_lat: int,
    ) -> torch.Tensor:
        B, _, dim = hidden.shape
        noisy = hidden[:, seq_len : 2 * seq_len]
        Hp = max(1, H_lat // 2)
        Wp = max(1, W_lat // 2)
        assert noisy.shape[1] == F_lat * Hp * Wp, (
            f"Hidden state slice {noisy.shape} cannot be reshaped to "
            f"({F_lat}, {Hp}, {Wp}, {dim})."
        )
        grid = noisy.view(B, F_lat, Hp, Wp, dim)
        grid = grid.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return grid[:, :, :H_lat, :W_lat, :]

    def _weighted_output_distill(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        weights: torch.Tensor,
        timestep: torch.Tensor,
        noise_shape: torch.Size,
    ) -> torch.Tensor:
        diff_sq = (student_pred.float() - teacher_pred.float()).pow(2).mean(dim=1)
        w = weights.to(diff_sq.dtype)

        num = (w * diff_sq).flatten(1).sum(dim=1)
        den = w.flatten(1).sum(dim=1).clamp_min(1e-6)
        per_sample = num / den

        if self.amd_config.distill_timestep_weighting:
            tw = (
                self.scheduler.training_weight(timestep.flatten(0, 1))
                .unflatten(0, (noise_shape[0], noise_shape[1]))
                .to(per_sample.device)
            )
            per_sample = per_sample * tw.mean(dim=1)

        return per_sample.mean()

    def _weight_regulariser(self, weights: torch.Tensor) -> torch.Tensor:
        reg = (
            (weights.mean() - self.tr_config.weight_mean_target).pow(2)
            * self.tr_config.weight_mean_reg_coeff
        )
        if self.tr_config.weight_entropy_coeff > 0.0:
            eps = 1e-6
            w_clamped = weights.clamp(eps, 1.0 - eps)
            entropy = -(
                w_clamped * torch.log(w_clamped)
                + (1.0 - w_clamped) * torch.log(1.0 - w_clamped)
            )
            reg = reg - self.tr_config.weight_entropy_coeff * entropy.mean()
        return reg

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        embodiment_id = action_input.embodiment_id
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask
        state_features = action_input.state
        actions = action_input.action

        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"

        videos = data["images"]
        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

        target_h = getattr(self.config, "target_video_height", None)
        target_w = getattr(self.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        latents = self.encode_video(
            videos,
            self.tiled,
            (self.tile_size_height, self.tile_size_width),
            (self.tile_stride_height, self.tile_stride_width),
        )

        _, _, num_frames_pix, height_pix, width_pix = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, ys, _ = self.encode_image(image, num_frames_pix, height_pix, width_pix)

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)

        noise = torch.randn_like(latents)
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        motion_prior = self._compute_motion_prior(latents.detach())

        if self.config.decouple_video_action_noise:
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        elif self.config.use_high_noise_emphasis:
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        else:
            timestep_id = torch.randint(
                0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1])
            )

        timestep_id_block = timestep_id[:, 1:].reshape(
            timestep_id.shape[0], -1, self.num_frame_per_block
        )
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert (
                actions.shape[1] / (noise.shape[1] - 1)
                == (self.model.num_action_per_block // self.num_frame_per_block)
            )
            assert (
                (noise.shape[1] - 1) / state_features.shape[1]
                == (self.num_frame_per_block // self.model.num_state_per_block)
            )
            if self.config.decouple_video_action_noise:
                timestep_action_id = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (actions.shape[0], actions.shape[1]),
                )
            else:
                timestep_action_id = timestep_id_block.repeat(
                    1, 1, actions.shape[1] // (noise.shape[1] - 1)
                )
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
        else:
            noise_action = None
            timestep_action_id = None

        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        tokens_per_frame = (height // 2) * (width // 2)
        seq_len = num_frames * tokens_per_frame

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(
                actions, noise_action, timestep_action
            )
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        autocast_ctx = torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self._device).type
        )

        run_teacher = (
            self.amd_config.enable_amd_distill
            and self.training
            and (
                self.amd_config.teacher_dropout_prob <= 0.0
                or torch.rand((), device="cpu").item() >= self.amd_config.teacher_dropout_prob
            )
        )

        want_hidden_distill = run_teacher and self.amd_config.distill_hidden_coeff > 0.0
        want_teacher_hidden_for_w = (
            run_teacher
            and self.tr_config.enable_token_weighting
            and self.tr_config.weight_predictor_use_teacher_hidden
        )
        capture_block = (
            self._select_block() if (want_hidden_distill or want_teacher_hidden_for_w) else None
        )

        with autocast_ctx:
            student_capture_ctx = (
                _HiddenStateCapture(capture_block) if want_hidden_distill else contextlib.nullcontext()
            )
            with student_capture_ctx as student_capture:
                if actions.numel() > 0:
                    video_noise_pred, action_noise_pred = self.model(
                        noisy_latents.transpose(1, 2),
                        timestep=timestep,
                        clip_feature=clip_feas,
                        y=ys,
                        context=prompt_embs,
                        seq_len=seq_len,
                        state=state_features,
                        embodiment_id=embodiment_id,
                        action=noisy_actions,
                        timestep_action=timestep_action,
                        clean_x=latents.transpose(1, 2),
                    )
                else:
                    video_noise_pred, action_noise_pred = self.model(
                        noisy_latents.transpose(1, 2),
                        timestep=timestep,
                        timestep_action=timestep_action,
                        clip_feature=clip_feas,
                        y=ys,
                        context=prompt_embs,
                        seq_len=seq_len,
                        state=state_features,
                        embodiment_id=embodiment_id,
                        clean_x=latents.transpose(1, 2),
                    )
            student_hidden = student_capture.hidden if want_hidden_distill else None

        teacher_video_pred = None
        teacher_hidden_distill = None
        teacher_hidden_for_w = None
        if run_teacher:
            with torch.no_grad():
                with _lora_disabled(self.model):
                    with autocast_ctx:
                        teacher_capture_ctx = (
                            _HiddenStateCapture(capture_block)
                            if (want_hidden_distill or want_teacher_hidden_for_w)
                            else contextlib.nullcontext()
                        )
                        with teacher_capture_ctx as teacher_capture:
                            teacher_video_pred, _ = self.model(
                                noisy_latents.transpose(1, 2),
                                timestep=timestep,
                                timestep_action=None,
                                clip_feature=clip_feas,
                                y=ys,
                                context=prompt_embs,
                                seq_len=seq_len,
                                state=None,
                                embodiment_id=None,
                                action=None,
                                clean_x=latents.transpose(1, 2),
                            )
                        teacher_hidden_capture = (
                            teacher_capture.hidden
                            if (want_hidden_distill or want_teacher_hidden_for_w)
                            else None
                        )
            if want_hidden_distill:
                teacher_hidden_distill = teacher_hidden_capture
            if want_teacher_hidden_for_w:
                teacher_hidden_for_w = teacher_hidden_capture

        with autocast_ctx:
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]

            dynamics_loss_per_sample = F.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction="none"
            ).mean(dim=(1, 3, 4))
            weight_dynamics = (
                dynamics_loss_per_sample
                * self.scheduler.training_weight(timestep.flatten(0, 1))
                .unflatten(0, (noise.shape[0], noise.shape[1]))
                .to(self._device)
            )
            weighted_dynamics_loss = weight_dynamics.mean()

            if actions.numel() > 0:
                action_loss_per_sample = (
                    F.mse_loss(
                        action_noise_pred.float(),
                        training_target_action.float(),
                        reduction="none",
                    )
                    * action_mask
                )
                action_loss_per_sample = (
                    has_real_action[:, None].float() * action_loss_per_sample
                )
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)

            distill_output_loss = torch.tensor(0.0, device=self._device)
            distill_hidden_loss = torch.tensor(0.0, device=self._device)
            weight_reg_loss = torch.tensor(0.0, device=self._device)
            mean_w_value = torch.tensor(0.0, device=self._device)

            if run_teacher:
                t_pred = teacher_video_pred
                if t_pred.shape != video_noise_pred.shape:
                    t_pred = t_pred[
                        ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                    ]

                F_lat = video_noise_pred.shape[2]
                H_lat = video_noise_pred.shape[3]
                W_lat = video_noise_pred.shape[4]
                motion = motion_prior[:, :F_lat, :H_lat, :W_lat]

                if self.tr_config.enable_token_weighting:
                    teacher_hidden_grid = None
                    if want_teacher_hidden_for_w and teacher_hidden_for_w is not None:
                        teacher_hidden_grid = self._hidden_to_grid(
                            teacher_hidden_for_w.detach(), seq_len, F_lat, H_lat, W_lat
                        )
                    weights = self.weight_predictor(motion, teacher_hidden_grid)
                    distill_output_loss = self._weighted_output_distill(
                        video_noise_pred, t_pred.detach(), weights, timestep, noise.shape
                    )
                    weight_reg_loss = self._weight_regulariser(weights)
                    mean_w_value = weights.detach().mean()
                else:
                    distill_output_loss = self._output_distill_loss(
                        video_noise_pred, t_pred.detach(), timestep, noise.shape
                    )

                if (
                    self.amd_config.distill_hidden_coeff > 0.0
                    and student_hidden is not None
                    and teacher_hidden_distill is not None
                ):
                    distill_hidden_loss = self._compute_hidden_distill(
                        student_hidden, teacher_hidden_distill, seq_len
                    )

            loss = weighted_dynamics_loss + weighted_action_loss
            if run_teacher:
                loss = (
                    loss
                    + self.amd_config.distill_output_coeff * distill_output_loss
                    + self.amd_config.distill_hidden_coeff * distill_hidden_loss
                    + weight_reg_loss
                )

        if not self._tr_logged and run_teacher:
            print(
                f"[AMD-TR] enable_token_weighting={self.tr_config.enable_token_weighting} "
                f"distill_output_coeff={self.amd_config.distill_output_coeff} "
                f"weight_mean_target={self.tr_config.weight_mean_target} "
                f"weight_mean_reg_coeff={self.tr_config.weight_mean_reg_coeff} "
                f"weight_entropy_coeff={self.tr_config.weight_entropy_coeff} "
                f"motion_prior_strength={self.tr_config.motion_prior_strength} "
                f"motion_prior_learnable={self.tr_config.motion_prior_learnable} "
                f"use_teacher_hidden={self.tr_config.weight_predictor_use_teacher_hidden}"
            )
            self._tr_logged = True

        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
            "distill_output_loss": distill_output_loss,
            "distill_hidden_loss": distill_hidden_loss,
            "weight_reg_loss": weight_reg_loss,
            "mean_distill_weight": mean_w_value,
        }

        return BatchFeature(data=output_dict)
