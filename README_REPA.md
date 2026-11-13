# Stage 3 self-forcing DMD + REPA (Cosmos3 Sim-Bimanual)

This fork of [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) (base commit `b2faa26`)
adds a REPA / VideoREPA alignment loss to the Sim-Bimanual Stage 3 self-forcing DMD distillation. Stages 0–2
are unchanged.

## What the REPA loss does

- **Features:** the student's generation tokens after **decoder layer 20** (of 36). They go through a new MLP
  projector, 4096 → 2048 → 2048 → 1024. The projector is part of the student only; the teacher and critic are
  unchanged.
- **Targets:** the paired real video, encoded by the frozen
  [OmniStream](https://huggingface.co/StreamFormer/OmniStream) encoder (ViT-L, 1024-dim):
  - frames are resized to 16 px per student token and normalized with ImageNet statistics;
  - the 4 pixel frames behind each latent frame are averaged.
- **Shapes** for a 121-frame 480×640 clip: 30 generated latent frames × (15×20 tokens) × 1024 on both sides.
  Latent frame 0 is the given first frame, so it is not aligned.
- **Loss:** token-relation distillation. Each frame's tokens are compared with all of the clip's tokens, and
  the penalty is `relu(|student − target| − 0.1)`, averaged. It is added to the DMD loss with weight **0.5**.
- **Where gradients come from:** each 2-latent-frame chunk is replayed once by the student with gradients.
  That replay computes the full-video loss with gradients through its own chunk only; the other chunks use the
  detached features recorded during the rollout. Summed over the chunks, this gives exactly the gradient of the
  full-video loss.

### Settings

All REPA settings are in `make_local_bimanual_stage3_repa_experiment` in
`cosmos_framework/configs/simulation/common/bimanual_profiles.py`:

| Setting | Value |
|---|---|
| `repa_loss_weight` | 0.5 |
| `repa_align_layer` | 20 (counted from 1) |
| `repa_projector_dim` | 2048 |
| `repa_align_dim` | 1024 |
| `repa_trd_margin` | 0.1 |
| `repa_encoder_path` | `$COSMOS_SIM_BIMANUAL_REPA_ENCODER` |

You can also override any of them on the command line, for example `model.config.repa_align_layer=18`.

### Code

| File | What it adds |
|---|---|
| `cosmos_framework/model/generator/distillation/repa/` | Vendored OmniStream encoder (`omnistream/`), projector, target encoding and the token-relation loss (`losses.py`) |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` | Projector and the layer capture (`repa_align_layer`, `repa_capture`, `_repa_features`) |
| `cosmos_framework/model/generator/distillation/self_forcing_dmd.py` | Config fields, rollout feature capture, targets (`_build_repa_context`), per-chunk loss (`_repa_replay_loss`), and student loading without a REPA head |
| `cosmos_framework/configs/simulation/...` | Experiment `sim_local_bimanual_stage3_sfdmd_repa` |
| `examples/toml/sft_config/sim_bimanual_stage3repa_sfdmd.toml` | TOML for that experiment |
| `examples/sim_bimanual_repa/train_stage3_repa.sh` | Launcher: 8 GPUs, global batch 16 |

## 1. Install

Set up the environment as described in [README.md](README.md):

```bash
uv sync --all-extras --group=cu130-train   # or --group=cu128-train
```

## 2. Download what Stage 3 needs

```bash
# OmniStream encoder (target features)
hf download StreamFormer/OmniStream --local-dir /path/to/OmniStream

# Wan2.2 VAE: the original Wan2.2_VAE.pth, not the diffusers copy
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir /path/to/wan22_vae
```

You also need two **training (DCP) checkpoints** from your own Stage 0 and Stage 2 runs. Pass each one's
iteration root, e.g. `.../checkpoints/iter_000035000`. The directory must contain `model/`.

- **Stage 0** loads into the teacher and the fake-score critic.
- **Stage 2** loads into the student. Its checkpoint has no REPA projector, so the projector starts from fresh
  initialization; the loader logs "no REPA align head; 6 keys keep their init".
- **Exported checkpoints don't work:** Hugging Face exports (safetensors) can't be used here.
- **EMA:** the loader reads the checkpoints' EMA weights (`net_ema`), so Stages 0–2 should have been trained
  with EMA on. That is the reference recipe.

## 3. Train (8 GPUs, global batch 16)

```bash
export BUNDLE=/path/to/bundle/train            # local Sim-Bimanual bundle (contains resources/tokenizer)
export STAGE0_CKPT=/path/to/stage0/checkpoints/iter_000035000
export STAGE2_CKPT=/path/to/stage2/checkpoints/iter_000004000
export VAE_PTH=/path/to/wan22_vae/Wan2.2_VAE.pth
export REPA_ENCODER=/path/to/OmniStream
export OUTPUT_ROOT=/path/to/runs

examples/sim_bimanual_repa/train_stage3_repa.sh
```

The launcher runs FSDP over 8 GPUs with no context parallelism, 1 sample per GPU and gradient accumulation 2,
so the global batch is 16. Other defaults:

- EMA on;
- 1,500 iterations, a checkpoint every 100;
- the Stage 3 recipe's learning rates: student 1e-6, critic 2e-7;
- a student update every 6th iteration; the iterations in between update the critic.

Optional environment variables: `GPUS`, `GLOBAL_BATCH`, `MAX_ITER`, `SAVE_ITER`, `RUN_NAME`, `EMA=0`.
Arguments after the script are Hydra overrides.

Every student iteration logs the REPA value:

```
Iteration N, student phase, REPA trd loss (full video): 0.11..., weight 0.5
```

In the trainer's loss outputs, `repa_trd_loss` is that same full-video value. `total_generator_loss` is the
VSD loss plus 0.5 × the REPA loss.

## Notes and limits

- **Training mode:** REPA works with replayed self-forcing training (`sf_replayed_training=True`, the Stage 3
  recipe) and the causal-AR rollout. With anything else, the config raises an error.
- **Real video in memory:** the real video must still be in memory at training time. That rules out context
  parallelism and per-camera VAE encoding; both raise an error.
- **Memory:**
  - Stage 3 holds three ~15B networks (student, teacher, critic), plus Adam states for the student and the
    critic.
  - On 4×H100 (95 GB) it ran out of memory, so use 8 GPUs.
  - If the critic rollout still runs out of memory, add `model.config.sf_keep_student_fsdp_unsharded_during_rollout=false`.
    It saves memory but makes critic iterations much slower: about 15 s → 94 s each in our 4-GPU test.
- **EMA off (`EMA=0`):**
  - Checkpoints trained without EMA have only `net` weights.
  - To load them, give the loader paths that end in `.dcp/model`: link `X.dcp -> <iteration root>` and pass
    `X.dcp` as `STAGE0_CKPT` / `STAGE2_CKPT`.
- **Projector learning rate:** the projector trains at the student's learning rate (1e-6). That is low for a
  freshly initialized layer; raise it with `optimizer.net.lr_multipliers` if needed.
- **Export:** `export_model.py` rebuilds the base student, which has no projector, so exported checkpoints
  should contain no REPA weights. This has not been tested.
- **Clip length:** a Stage 3 rollout spans the whole training clip. Use clips of about 121 frames, the length
  Stages 1–3 use.
- **Local patch:** `cosmos_framework/model/generator/mot/causal_attention.py` carries a fix that skips packed
  segments shorter than one action supertoken in the null-action zeroing. Without it, Stage 1+ training crashed
  on our data.

## Test status

- **Smoke test** on 4 GPUs with the 2-clip test bundle:
  - checkpoints loaded with the REPA head freshly initialized;
  - a full student update with REPA completed (loss 0.111);
  - critic updates ran.
- **Memory:** the second student update ran out of memory on 4 GPUs.
- **Not yet run:** an 8-GPU run, and anything beyond the smoke test.
