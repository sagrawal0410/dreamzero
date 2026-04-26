#!/bin/bash
# Usage:
#   bash scripts/train/droid_training_wan22_amd.sh

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "/root/yejink/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/yejink/dreamzero
elif [ -d "/root/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/dreamzero
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-8}
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"$DREAMZERO_ROOT/data/droid_lerobot"}
if [ "$DROID_DATA_ROOT" = "./data/droid_lerobot" ]; then
    DROID_DATA_ROOT="$DREAMZERO_ROOT/data/droid_lerobot"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_droid_wan22_amd_lora"}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}

DISTILL_OUTPUT_COEFF=${DISTILL_OUTPUT_COEFF:-1.0}
DISTILL_HIDDEN_COEFF=${DISTILL_HIDDEN_COEFF:-0.0}
DISTILL_HIDDEN_LAYER=${DISTILL_HIDDEN_LAYER:--1}
DISTILL_HIDDEN_TARGET=${DISTILL_HIDDEN_TARGET:-noisy}
TEACHER_DROPOUT_PROB=${TEACHER_DROPOUT_PROB:-0.0}

if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (for CLIP only)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi

if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    echo "Download with: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DROID_DATA_ROOT"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi
PYTHON_311="/usr/bin/python3.11"
if [ -x "$PYTHON_311" ]; then
    if [ -n "${FIX_NUMPY_IN_SCRIPT:-}" ]; then
        "$PYTHON_311" -m pip install "numpy==1.26.4" --force-reinstall -q 2>/dev/null || true
    fi
    RUN_CMD=( "$PYTHON_311" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using image Python 3.11: $PYTHON_311"
else
    RUN_CMD=( python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using: $(command -v python3)"
fi
cd "$DREAMZERO_ROOT"

"${RUN_CMD[@]}" \
    report_to=wandb \
    data=dreamzero/droid_relative_wan22 \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_amd_distill \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=100 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    save_lora_only=true \
    max_chunk_size=4 \
    save_strategy=no \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    action_head_cfg.config.distill_output_coeff=$DISTILL_OUTPUT_COEFF \
    action_head_cfg.config.distill_hidden_coeff=$DISTILL_HIDDEN_COEFF \
    action_head_cfg.config.distill_hidden_layer=$DISTILL_HIDDEN_LAYER \
    action_head_cfg.config.distill_hidden_target=$DISTILL_HIDDEN_TARGET \
    action_head_cfg.config.teacher_dropout_prob=$TEACHER_DROPOUT_PROB
