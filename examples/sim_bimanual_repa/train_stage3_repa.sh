#!/usr/bin/env bash
# Stage 3 (self-forcing DMD) + REPA on one node. See README_REPA.md.
# Default: 8 GPUs, FSDP 8 x CP 1, 1 sample per GPU, gradient accumulation 2 -> global batch 16, EMA on.
#
# Usage: examples/sim_bimanual_repa/train_stage3_repa.sh [--dryrun] [extra hydra overrides...]
# Required env:
#   BUNDLE        local Sim-Bimanual bundle train/ dir
#   STAGE0_CKPT   Stage 0 checkpoint iteration root (teacher + fake-score critic), e.g. .../checkpoints/iter_000035000
#   STAGE2_CKPT   Stage 2 checkpoint iteration root (student init)
#   VAE_PTH       Wan2.2_VAE.pth
#   REPA_ENCODER  OmniStream checkpoint dir (hf download StreamFormer/OmniStream)
#   OUTPUT_ROOT   checkpoint/log root
# Optional env: GPUS (0,...,7), GLOBAL_BATCH (16), MAX_ITER (1500), SAVE_ITER (100), RUN_NAME, EMA (1|0, default 1)
set -euo pipefail
: "${BUNDLE:?}" "${STAGE0_CKPT:?}" "${STAGE2_CKPT:?}" "${VAE_PTH:?}" "${REPA_ENCODER:?}" "${OUTPUT_ROOT:?}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(echo "$GPUS" | tr , '\n' | wc -l)
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
(( GLOBAL_BATCH % NGPU == 0 )) || { echo "GLOBAL_BATCH ($GLOBAL_BATCH) must be divisible by the GPU count ($NGPU)"; exit 1; }
ACCUM=$(( GLOBAL_BATCH / NGPU ))  # 1 sample per GPU per micro-step

cd "$REPO"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE=$BUNDLE
export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT=$STAGE0_CKPT
export COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT=$STAGE2_CKPT
export COSMOS_SIM_BIMANUAL_REPA_ENCODER=$REPA_ENCODER
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT
export CUDA_VISIBLE_DEVICES=$GPUS
TOML=examples/toml/sft_config/sim_bimanual_stage3repa_sfdmd.toml

OVR=(
  "job.name=${RUN_NAME:-stage3-sfdmd-repa}"
  "model.config.tokenizer.vae_path=$VAE_PTH"
  "model.config.parallelism.data_parallel_shard_degree=$NGPU"
  model.config.parallelism.context_parallel_shard_degree=1
  model_parallel.context_parallel_size=1
  dataloader_train.max_samples_per_batch=1
  "trainer.grad_accum_iter=$ACCUM"
  "trainer.max_iter=${MAX_ITER:-1500}"
  "checkpoint.save_iter=${SAVE_ITER:-100}"
  trainer.logging_iter=10
)
# EMA off keeps one fp32 copy less; the Stage 0/2 checkpoints must then be loaded from their "net" weights
# (see README_REPA.md, "EMA off").
[[ "${EMA:-1}" == "0" ]] && OVR+=(model.config.ema.enabled=false)
DRY=()
if [[ "${1:-}" == "--dryrun" ]]; then DRY=(--dryrun); shift; fi
OVR+=("$@")

echo "TOML=$TOML GPUS=$GPUS global_batch=$GLOBAL_BATCH (1 x $NGPU GPUs x accum $ACCUM) EMA=${EMA:-1}"
if [[ ${#DRY[@]} -gt 0 ]]; then
  python -m cosmos_framework.scripts.train --sft-toml="$TOML" --dryrun -- "${OVR[@]}"
else
  torchrun --standalone --nproc-per-node="$NGPU" -m cosmos_framework.scripts.train --sft-toml="$TOML" -- "${OVR[@]}"
fi
