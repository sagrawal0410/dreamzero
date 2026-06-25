#!/bin/bash
# CALA-WAM experiment pipeline: Stage 0 setup through Stage 5 allocator eval.
#
# Usage:
#   CHECKPOINT=/path/to/DreamZero-DROID DATASET=/path/to/droid_lerobot \
#     bash scripts/run_cala_wam_pipeline.sh
#
# Smoke run (fewer examples):
#   SMOKE=1 bash scripts/run_cala_wam_pipeline.sh
#
# Run one stage only:
#   STAGE=3 bash scripts/run_cala_wam_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/checkpoints/DreamZero-DROID}"
DATASET="${DATASET:-$REPO_ROOT/data/droid_lerobot}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs/cala_wam}"
STAGE="${STAGE:-all}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  EPISODES_PER_GROUP=2
  TIMESTEPS=3
  NUM_EXAMPLES=2
  NUM_EPISODES=2
  EPOCHS=5
else
  EPISODES_PER_GROUP=8
  TIMESTEPS=5
  NUM_EXAMPLES=8
  NUM_EPISODES=6
  EPOCHS=60
fi

SUITE_DIR="$RUNS_DIR/stage0_suite"
MAPS_DIR="$RUNS_DIR/stage1_maps"
SALIENCY_DIR="$RUNS_DIR/stage2_saliency"
ALLOC_DIR="$RUNS_DIR/stage3_alloc"
CLOSED_DIR="$RUNS_DIR/stage4_closed_loop"
ALLOCATOR_DIR="$RUNS_DIR/stage5_allocator"

run_stage() {
  local n="$1"
  shift
  if [[ "$STAGE" != "all" && "$STAGE" != "$n" ]]; then
    return 0
  fi
  echo "=== Stage $n ==="
  "$@"
}

run_stage 0 python "$REPO_ROOT/scripts/stage0/build_task_suite.py" \
  --dataset_root "$DATASET" \
  --task_groups "$REPO_ROOT/scripts/stage0/configs/task_groups.yaml" \
  --num_episodes_per_group "$EPISODES_PER_GROUP" \
  --num_timesteps_per_episode "$TIMESTEPS" \
  --output_dir "$SUITE_DIR"

run_stage 0 python "$REPO_ROOT/scripts/stage0/eval_baseline.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --num_seeds 3 \
  --output_dir "$RUNS_DIR/stage0_baseline"

run_stage 0 python "$REPO_ROOT/scripts/stage0/stability_analysis.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --num_examples "$NUM_EXAMPLES" \
  --output_dir "$RUNS_DIR/stage0_stability"

run_stage 1 python "$REPO_ROOT/scripts/stage1/causal_maps_compute.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --num_examples "$NUM_EXAMPLES" \
  --output_dir "$MAPS_DIR"

run_stage 1 python "$REPO_ROOT/scripts/stage1/causal_maps_analyze.py" \
  --maps_dir "$MAPS_DIR" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --noise_floor "$RUNS_DIR/stage0_stability/noise_floor.json" \
  --output_dir "$RUNS_DIR/stage1_analysis"

run_stage 2 python "$REPO_ROOT/scripts/stage2/saliency_compute.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --num_examples "$NUM_EXAMPLES" \
  --output_dir "$SALIENCY_DIR"

run_stage 2 python "$REPO_ROOT/scripts/stage2/saliency_analyze.py" \
  --maps_dir "$MAPS_DIR" \
  --saliency_dir "$SALIENCY_DIR" \
  --output_dir "$RUNS_DIR/stage2_analysis"

run_stage 3 python "$REPO_ROOT/scripts/stage3/allocation_compute.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --maps_dir "$MAPS_DIR" \
  --saliency_dir "$SALIENCY_DIR" \
  --num_examples "$NUM_EXAMPLES" \
  --output_dir "$ALLOC_DIR"

run_stage 3 python "$REPO_ROOT/scripts/stage3/allocation_analyze.py" \
  --rows "$ALLOC_DIR/all_rows.csv" \
  --output_dir "$RUNS_DIR/stage3_analysis"

run_stage 4 python "$REPO_ROOT/scripts/stage4/closed_loop_compute.py" \
  --checkpoint "$CHECKPOINT" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --maps_dir "$MAPS_DIR" \
  --saliency_dir "$SALIENCY_DIR" \
  --num_episodes "$NUM_EPISODES" \
  --output_dir "$CLOSED_DIR"

run_stage 4 python "$REPO_ROOT/scripts/stage4/closed_loop_analyze.py" \
  --rows "$CLOSED_DIR/all_rows.csv" \
  --maps_dir "$MAPS_DIR" \
  --saliency_dir "$SALIENCY_DIR" \
  --output_dir "$RUNS_DIR/stage4_analysis"

run_stage 5 python "$REPO_ROOT/scripts/stage5/train_allocator.py" \
  --maps_dir "$MAPS_DIR" \
  --task_suite "$SUITE_DIR/manifest.json" \
  --epochs "$EPOCHS" \
  --output_dir "$ALLOCATOR_DIR"

run_stage 5 python "$REPO_ROOT/scripts/stage5/analyze_allocator.py" \
  --maps_dir "$MAPS_DIR" \
  --runs "$ALLOCATOR_DIR" \
  --output_dir "$RUNS_DIR/stage5_analysis"

echo "Done. Artifacts under $RUNS_DIR"
