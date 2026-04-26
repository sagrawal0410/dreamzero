from dataclasses import dataclass, field
import contextlib
import logging
from typing import Optional

from einops import rearrange
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
    WANPolicyHeadConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class WANPolicyHeadAMDConfig(WANPolicyHeadConfig):
    enable_amd_distill: bool = field(
        default=True,
        metadata={"help": "Run the frozen-teacher pass and add distillation losses."},
    )
    distill_output_coeff: float = field(
        default=1.0,
        metadata={"help": "Weight on the video-noise-prediction (output) distillation loss."},
    )
    distill_hidden_coeff: float = field(
        default=0.0,
        metadata={"help": "Weight on intermediate hidden-state distillation. 0 disables."},
    )
    distill_hidden_layer: int = field(
        default=-1,
        metadata={
            "help": "DiT block index whose hidden state is matched. Negative indices count "
            "from the last block. Only used when distill_hidden_coeff > 0."
        },
    )
    distill_hidden_target: str = field(
        default="noisy",
        metadata={
            "help": "Which slice of the DiT hidden state to distill on: 'noisy' (the "
            "denoising half), 'clean' (the clean-context half), or 'both'. Only "
            "applies to hidden-state distillation."
        },
    )
    distill_timestep_weighting: bool = field(
        default=True,
        metadata={
            "help": "Multiply the per-sample distillation loss by the flow-matching "
            "timestep weight, mirroring the dynamics loss."
        },
    )
    teacher_dropout_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of skipping the teacher pass on a given step. Useful for "
            "warming up or amortising compute."
        },
    )
    require_lora: bool = field(
        default=True,
        metadata={
            "help": "If True, raise unless the student is wrapped with LoRA so the teacher "
            "can be obtained for free by disabling adapters."
        },
    )


def _lora_disabled(model: torch.nn.Module):
    if hasattr(model, "disable_adapter") and callable(model.disable_adapter):
        return model.disable_adapter()
    return contextlib.nullcontext()


def _get_dit_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    return base.blocks


class _HiddenStateCapture:
    def __init__(self, block: torch.nn.Module):
        self._block = block
        self._handle = None
        self.hidden: Optional[torch.Tensor] = None

    def __enter__(self):
        self.hidden = None

        def _hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.hidden = tensor

        self._handle = self._block.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class WANPolicyHeadAMD(WANPolicyHead):
    config_class = WANPolicyHeadAMDConfig

    def __init__(self, config: WANPolicyHeadAMDConfig):
        super().__init__(config)
        self.amd_config: WANPolicyHeadAMDConfig = config

        if (
            getattr(config, "enable_amd_distill", False)
            and getattr(config, "require_lora", True)
            and self.train_architecture != "lora"
        ):
            raise ValueError(
                "WANPolicyHeadAMD with require_lora=True only supports train_architecture='lora'. "
                "Disable require_lora or switch to LoRA training."
            )

        self._distill_logged = False

    def _select_block(self) -> torch.nn.Module:
        blocks = _get_dit_blocks(self.model)
        idx = self.amd_config.distill_hidden_layer
        if idx < 0:
            idx = len(blocks) + idx
        idx = max(0, min(len(blocks) - 1, idx))
        return blocks[idx]

    def _compute_hidden_distill(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        target = self.amd_config.distill_hidden_target
        if target == "clean":
            s = student_hidden[:, :seq_len]
            t = teacher_hidden[:, :seq_len]
        elif target == "noisy":
            s = student_hidden[:, seq_len : 2 * seq_len]
            t = teacher_hidden[:, seq_len : 2 * seq_len]
        elif target == "both":
            s = student_hidden[:, : 2 * seq_len]
            t = teacher_hidden[:, : 2 * seq_len]
        else:
            raise ValueError(f"Unknown distill_hidden_target={target!r}")
        return F.mse_loss(s.float(), t.detach().float())

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
            ), (
                f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, "
                f"video.shape, {videos.shape}, latents.shape, {latents.shape}"
            )
            assert (
                (noise.shape[1] - 1) / state_features.shape[1]
                == (self.num_frame_per_block // self.model.num_state_per_block)
            ), (
                f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, "
                f"video.shape, {videos.shape}, latents.shape, {latents.shape}"
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

        want_hidden = run_teacher and self.amd_config.distill_hidden_coeff > 0.0
        capture_block = self._select_block() if want_hidden else None

        with autocast_ctx:
            student_capture_ctx = (
                _HiddenStateCapture(capture_block) if want_hidden else contextlib.nullcontext()
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
            student_hidden = student_capture.hidden if want_hidden else None

        teacher_video_pred = None
        teacher_hidden = None
        if run_teacher:
            with torch.no_grad():
                with _lora_disabled(self.model):
                    with autocast_ctx:
                        teacher_capture_ctx = (
                            _HiddenStateCapture(capture_block)
                            if want_hidden
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
                        teacher_hidden = teacher_capture.hidden if want_hidden else None

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
            if run_teacher:
                t_pred = teacher_video_pred
                if t_pred.shape != video_noise_pred.shape:
                    t_pred = t_pred[
                        ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                    ]
                distill_output_loss = self._output_distill_loss(
                    video_noise_pred, t_pred.detach(), timestep, noise.shape
                )

                if want_hidden and student_hidden is not None and teacher_hidden is not None:
                    distill_hidden_loss = self._compute_hidden_distill(
                        student_hidden, teacher_hidden, seq_len
                    )

            loss = weighted_dynamics_loss + weighted_action_loss
            if run_teacher:
                loss = (
                    loss
                    + self.amd_config.distill_output_coeff * distill_output_loss
                    + self.amd_config.distill_hidden_coeff * distill_hidden_loss
                )

        if not self._distill_logged and run_teacher:
            print(
                f"[AMD] enable_amd_distill={self.amd_config.enable_amd_distill} "
                f"distill_output_coeff={self.amd_config.distill_output_coeff} "
                f"distill_hidden_coeff={self.amd_config.distill_hidden_coeff} "
                f"distill_hidden_layer={self.amd_config.distill_hidden_layer} "
                f"distill_hidden_target={self.amd_config.distill_hidden_target} "
                f"teacher_dropout_prob={self.amd_config.teacher_dropout_prob}"
            )
            self._distill_logged = True

        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
            "distill_output_loss": distill_output_loss,
            "distill_hidden_loss": distill_hidden_loss,
        }

        return BatchFeature(data=output_dict)

    def _output_distill_loss(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        timestep: torch.Tensor,
        noise_shape: torch.Size,
    ) -> torch.Tensor:
        diff = (student_pred.float() - teacher_pred.float()).pow(2)
        per_sample = diff.mean(dim=(1, 3, 4))
        if self.amd_config.distill_timestep_weighting:
            weight = (
                self.scheduler.training_weight(timestep.flatten(0, 1))
                .unflatten(0, (noise_shape[0], noise_shape[1]))
                .to(per_sample.device)
            )
            per_sample = per_sample * weight
        return per_sample.mean()
