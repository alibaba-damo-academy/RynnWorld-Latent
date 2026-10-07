# RynnWorld-Latent

**RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions**

Haoyu Zhao\*, Yinzhou Tang\*, Zixuan Wang\*, Xingyue Zhao, Qin Zhao, Kehan Li, Siteng Huang†, Xin Li, Zhongyu Li†
_DAMO Academy, Alibaba Group · Hong Kong Embodied AI Lab · CUHK · Hupan Lab_ (\* core contributors, † corresponding)

| | |
|---|---|
| 🏠 Homepage | https://alibaba-damo-academy.github.io/RynnWorld-Latent.github.io |
| 💻 GitHub | https://github.com/alibaba-damo-academy/RynnWorld-Latent |
| 🤗 Hugging Face | https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent |
| 🎯 ModelScope | https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-Latent |

---

## Overview

Simulating action outcomes across diverse environments is pivotal for scaling generalist agents, but modeling world dynamics for dexterous robotics is hard: embodiment interfaces are heterogeneous, robot data coverage is limited, and fine-grained action labels are scarce.

**RynnWorld-Latent** is a cross-embodiment video world model built for both **future prediction** and **scalable robot data generation**. It addresses the action-label bottleneck by using **continuous latent actions** as a unified representation of world-state transitions: each latent action compactly describes *what changed* between two observations, so human egocentric video and robot data can be modeled jointly **without embodiment-specific control commands**. Built on a pretrained video diffusion transformer, RynnWorld-Latent is **text-free** and adds **explicit frame-aligned action conditioning**, giving precise control over future dynamics while preserving the pretrained video prior. It is trained on a large cross-embodiment mixture (32 datasets, >265M frames) and post-trained on small-scale target-robot data to align the latent-action space with the target robot's control interface.

Beyond prediction, the model is a **data-generation engine** in two modes:
- **Interactive digital teleoperation** — the post-trained model is conditioned on target-robot actions, so operator control signals drive video generation.
- **Data synthesis** — conditioned on latent actions: given a source video, edit its first frame to swap the embodiment or environment, reuse the source video's latent actions, and roll out a new video whose interaction dynamics follow the source but appear under the new embodiment/environment.

> **Why latent actions.** Real action labels are scarce and embodiment-specific. The bundled **RynnLAM** distils a *motion-specific* continuous code from raw frame pairs (flow-reconstruction supervised), shared across humans and robots, so unlabeled video can drive action-conditioned training. RynnWorld-Latent is, to our knowledge, the first world model conditioned on a single action representation shared by humans and multiple robots.

**This repository** ships the world-model code needed to **understand the architecture, resume training, and run training + evaluation on a single node** reading the bundled local data. It is built on **NVIDIA Cosmos3-Edge** (3.37B) via the bundled `cosmos-framework`, and conditioned on latent actions from **RynnLAM** (bundled under [`rynnlam/`](rynnlam)).

---

## Model architecture

Full-parameter fine-tuning of NVIDIA **Cosmos3-Edge** (3.37B, Nemotron-2B backbone) into a **latent-action-conditioned forward-dynamics world model**:

> given a **first-frame observation** + **20 actions**, predict the **next 80 video frames**.

The actions are **RynnLAM** latent labels — a 608-dim continuous code, **not physical joint quantities** — which is what lets one conditioning interface serve human and multiple robot embodiments. The action-conditioning scheme is codenamed **film** (frame-aligned injection + two-tower encoder + per-frame FiLM).

### Dataflow

```
source video 81 frames ─▶ [Wan2.2 VAE, 16x spatial / 4x temporal] ─▶ 21 latent frames ─▶ [vae2llm] ─▶ vision tokens ─┐
caption                ─▶ [text tokenizer]                        ─▶ text tokens                                     │
                                                                                                                      │
actions 20x608         ─▶ [TwoTowerActionEncoder] ─▶ action tokens ─┬─▶ (1) frame-aligned additive injection ────────┤
   (k_tokens 512 + z 64 | cam 32)                                   └─▶ (2) per-frame FiLM  v*(1+gamma)+beta          │
                                                                                                                      ▼
                                              Cosmos3 shared backbone (MoT, joint attention, two_way)
                                                          ├─▶ [llm2vae]    -> video latent (rectified-flow trained)
                                                          └─▶ [llm2action] -> untrained under forward_dynamics
```

Under film, `RYNNWORLD_TEXT_FREE=1` empties the caption and disables every metadata augmentor, so the only conditioning left is **first frame + action**.

### Components

| Component | Notes |
|---|---|
| **Backbone** | Nemotron-2B dense LM (hidden 2048, 28 layers) + SigLIP2 vision tower + PatchMerger + mRoPE |
| **Paradigm** | Mixture-of-Transformers (MoT): understanding / generation experts share the backbone, joint attention (`two_way`); action tokens enter self-attention alongside vision tokens |
| **Video tokenizer** | Wan2.2 VAE, 16x spatial / 4x temporal compression, **frozen** |
| **Generation heads** | `vae2llm` / `llm2vae` (video latent ↔ hidden), `time_embedder` (diffusion timestep embedding) |
| **Action pathway** | `action2llm` = `TwoTowerActionEncoder` (replaces the stock `DomainAwareLinear` when `RYNNWORLD_ACTION_TOWER_SPLIT=1`), plus the `frame_gate` / `frame_film` injection parameters. `llm2action` and `action_modality_embed` stay stock |
| **Text tokenizer** | Loaded from `third_party/cosmos_tokenizers/edge/text_tokenizer/` (bundled, OpenMDW-1.1) instead of a HF Hub download; override with `RYNNWORLD_EDGE_HF_DIR` |
| **Objective** | Rectified Flow (flow matching) on video latents, `loss_scale=10.0`; `action_loss_weight=10.0` is inert under forward_dynamics because actions are never predicted |
| **Precision / parallelism** | bfloat16; FSDP `data_parallel_shard_degree=8`, `replicate=1`; activation checkpointing `mode="full"` |
| **torch.compile** | Disabled in every recipe TOML. `compiled_region` is **not** a valid TOML key (`CompileConfig` allows only `enabled` / `compile_dynamic`), so compile scope is decided attrs-side (default `"language"`). To try it, override `model.compile.enabled=true` per run and benchmark on your own image first |

### Key dimensions (`rynnworld_latent/dataset.py`, `action_conditioning.py`)

- **Action vector = 608 dims** = `ACTION_DIM`, laid out `[k_tokens 512 | z 64 | camera 32]`; written into the model's `max_action_dim`.
- **Two-tower split = `HAND_DIM`** (env `RYNNWORLD_ACTION_HAND_DIM`, default **576**): `x[..., :576]` is the k_tokens+z tower, `x[..., 576:]` the 32-dim camera tower.
- **Video chunk = 81 frames** = 1 conditioning + 80 predicted; 4x temporal VAE compression → **21 latent frames = 1 conditioning + 20 predicted**.
- **Actions per chunk = 20**, one per 4 RGB frames, aligned **1:1** with the 20 predicted latent frames — the premise the frame-aligned injection rests on.
- **DOMAIN_ID = 3** — Cosmos3-Edge's `hand_pose` action domain.

---

## Action conditioning ("film")

All of it lives in `rynnworld_latent/action_conditioning.py`, monkey-patched onto `Cosmos3VFMNetwork` so upstream cosmos source is never edited. **Training and inference must register the same patches** (both `scripts/train.py` and `rynnworld_latent/inference.py` call `register_action_conditioning_patches()`), otherwise `frame_gate` / `frame_film` are absent and checkpoint loading reports missing keys.

Stock action injection collapses to **action-blind, near-still** rollouts, and scaling the backbone does not fix it. film instead uses:

- **Two-tower action encoder** (`TwoTowerActionEncoder`) — non-camera tower `DomainAwareLinear(576→h)` + camera tower `DomainAwareLinear(32→h)`, summed, then a shared residual MLP `h + mlp(h)`. Camera tower and the MLP's second layer are **zero-initialised**, so at step 0 the added path is the identity. Each modality gets its own nonlinear transform and can be dropped/guided independently. (This is *not* "splitting one Linear in half", which is a mathematical no-op.)
- **Frame-aligned additive injection** — action step *j* covers rgb frames `[4j, 4j+4]` = exactly latent frame *j+1*; after the stock scatter, `packed_sequence[tgt] += frame_gate * action_token` over every spatial token of that frame. `frame_gate` is a per-channel `nn.Parameter(zeros(h))` (ControlNet-style identity at step 0). Covers predicted frames 1..T-1, excludes conditioning frame 0; only fires when action and vision token counts match.
- **Per-frame FiLM** (`frame_film = Linear(h→2h)`) — each action token produces that frame's `(gamma, beta)`, applied as `v*(1+gamma)+beta`; zero-initialised. Shares the frame-alignment index, so FiLM depends on frame injection.
- **Action CFG dropout + inference guidance** — training zeroes jointly / k-z-only / camera-only (`RYNNWORLD_ACTION_CFG_DROPOUT{,_HAND,_CAM}`, film uses 0.1). Inference rewrites `_run_classifier_free_guidance` so the two branches differ **only** in the action → pure action guidance.
- **text-free** — `RYNNWORLD_TEXT_FREE=1` → `cfg_dropout_rate=1.0` (empties caption) + disables the four metadata augmentors. cosmos's `cfg_dropout_rate` drops only the caption, never the action — hence the separate action-dropout mechanism.
- **condition forcing** — with prob `RYNNWORLD_COND_FORCE=0.3`, raise the sample's vision noise `sigma` to ≥ `RYNNWORLD_COND_FORCE_FLOOR=0.75` and recompute the timestep, closing the "at low noise just copy frame 0" shortcut. Training only.

**Fail-fast switch dependencies** (`register_action_conditioning_patches()` raises up front): action-CFG dropout requires `RYNNWORLD_ACTION_TOWER_SPLIT=1`; `RYNNWORLD_ACTION_FILM=1` requires `RYNNWORLD_ACTION_FRAME_INJECT=1`.

The full film switch set (used in the Training section below):
`CFG_DROPOUT=CFG_DROPOUT_HAND=CFG_DROPOUT_CAM=0.1`, `TEXT_FREE=1`, `COND_FORCE=0.3`, `COND_FORCE_FLOOR=0.75`, `FRAME_INJECT=TOWER_SPLIT=FILM=1`.

> **Evaluating a film checkpoint requires exporting the same switch set**, or the model structure will be missing `frame_gate` / `frame_film`. For a **DCP** checkpoint `scripts/inference/rollout.py` loads EMA weights by default — pass `--no-ema` for a run that did not save an EMA. For a **safetensors** checkpoint (e.g. the released weights in [Inference & evaluation](#inference--evaluation)) `--no-ema` is *required*: the direct loader reads the `net.*` weights and does not map the `net_ema.*` mirrors (EMA-to-regular loading is a DCP-only path).

---

## Install

```bash
git clone https://github.com/alibaba-damo-academy/RynnWorld-Latent && cd RynnWorld-Latent
python -m venv .venv && source .venv/bin/activate    # Python >= 3.10
pip install -e ".[video-sources]"                     # or: pip install -r requirements.txt

# cosmos-framework is vendored, not on PyPI — put it (and the repo) on PYTHONPATH:
export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
```

`requirements.txt` is the verified import closure of the vendored cosmos-framework (config load + video-SFT dataloader + training + inference) plus RynnLAM; `pyproject.toml` is authoritative and the two are kept in sync. **Pins matter:** `transformers>=4.57,<5` (5.x refactors `qwen3_vl` and breaks the cosmos processors), `torch>=2.7,<2.11` + `torchvision>=0.22,<0.26` (2.7 is the floor because the base→DCP converter imports `torch.distributed.checkpoint.hf_storage`, native only in ≥2.7; torch 2.11+'s dynamo rejects cosmos's forward), `wandb<0.28.2` (0.28.2 dropped `wandb.util.generate_id`), `numpy<2`. `huggingface_hub` and `tqdm` are **not** declared as direct dependencies (third-party SCA compliance): both arrive transitively via `transformers`/`datasets`, and `transformers>=4.57,<5` keeps `huggingface_hub` on the 0.x API cosmos needs; the vendored code loads `tqdm` dynamically with a no-op fallback (see `NOTICE`). For non-file video sources (zarr / hdf5) the `video-sources` extra adds matched `zarr>=3` + `numcodecs>=0.16` + `h5py`; plain mp4/webm needs only `av`. NVIDIA-image-only / optional pieces (`apex`, `flash_attn`/`natten`/`transformer_engine`, `multistorageclient`, `ray`, `lerobot`) are guarded in code or patched around by `scripts/train.py` (SDPA varlen fallback, inert lerobot stub) and are **not** required.

## Weights (gated)

```bash
huggingface-cli login                 # or: export HF_TOKEN=hf_...
bash scripts/setup/download_weights.sh
```

This downloads `nvidia/Cosmos3-Edge` (+ `Wan2.2_VAE.pth`), converts the backbone to the DCP layout cosmos trains on (via `scripts/checkpoints/convert_edge_to_dcp.py`), and prints:

```bash
export BASE_CHECKPOINT_PATH=/.../weights/Cosmos3-Edge-dcp
export WAN_VAE_PATH=/.../weights/Cosmos3-Edge/Wan2.2_VAE.pth
```

Request access to the gated repos on their Hugging Face pages first. Weight licenses: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). **Code plus three bundled sample chunks — no weights are redistributed in this repo.** The fine-tuned **RynnWorld-Latent** film checkpoint (what you get after training on top of this base) *is* published publicly on Hugging Face / ModelScope — see [Inference & evaluation](#inference--evaluation) to download and roll it out directly.

---

## Quickstart (single node, bundled data)

`data/` ships three real sample chunks (videos + RynnLAM latents + a v2 manifest) so a fresh clone trains and evaluates out of the box. With `BASE_CHECKPOINT_PATH` + `WAN_VAE_PATH` exported (above):

```bash
bash scripts/train_8gpu.sh          # quickstart: NGPU=1, bundled data/, a few iters
```

With no `MANIFEST_DIR` set, `train_8gpu.sh` trains on the bundled `data/manifest` using the single-GPU smoke recipe (`configs/examples/edge_manifest_local.toml`); override the GPU count / iterations with `QUICKSTART_NGPU` / `QUICKSTART_ITERS`. Set `MANIFEST_DIR` to train on a different manifest instead.

> **A run always writes one full checkpoint when it ends** (~20 GB for Edge), even a 3-iteration smoke: cosmos's trainer saves whenever `iteration % save_iter != 0`, so `checkpoint.save_iter` cannot suppress it. The quickstart therefore points `IMAGINAIRE_OUTPUT_ROOT` at local disk (`$TMPDIR/rynnworld_latent_quickstart`) by default. On a network mount that write can take many minutes *after* `Done with training.` is printed, which looks exactly like a hang. Set `IMAGINAIRE_OUTPUT_ROOT` explicitly if you want the artifact kept.

### Verifying an install

`scripts/test_world_model.sh` stages the whole pipeline on the bundled data; each stage is independently runnable and CPU-only unless noted:

```bash
bash scripts/test_world_model.sh --stage data   # manifest + latents: shapes, normalization, domain id
bash scripts/test_world_model.sh --stage lam    # re-encode videos with RynnLAM, diff vs the shipped latents (GPU)
bash scripts/test_world_model.sh --stage train-smoke   # 3 iters, proves the patches + DCP load (GPU)
bash scripts/test_world_model.sh --stage train         # 50 iters, loss should move (GPU)
bash scripts/test_world_model.sh --stage infer --checkpoint /path/to/iter_NNNNNNN  # rollout vs GT (GPU)
bash scripts/test_world_model.sh --stage all
```

It exports the full film switch set (override with `--no-film`) and writes to `$OUT_ROOT` (default `$TMPDIR/rynnworld_latent_checks`), never into the repo. `scripts/test_lam_inference.py` is the RynnLAM half on its own: it re-encodes `data/videos/*.mp4` and compares against `data/latents/*/latent.npz`, checking the `checkpoint_sha256` provenance, per-slice (`k`/`z`/`cam`) error and per-row cosine.

### Bundled samples (provenance)

Three chunks (81 frames / 20 latent actions each) copied verbatim from an in-house bimanual-teleop collection (`tianji_wuji_data`, `Bimanual_Lift` episodes 000028 / 000063 / 000116):

| file | source episode | frame window |
|---|---|---|
| `data/videos/tianji_wuji_data_episode_000028.mp4` | `Bimanual_Lift/episode_000028` | 160-240 |
| `data/videos/tianji_wuji_data_episode_000063.mp4` | `Bimanual_Lift/episode_000063` | 80-160 |
| `data/videos/tianji_wuji_data_episode_000116.mp4` | `Bimanual_Lift/episode_000116` | 0-80 |

`data/latents/<episode>/latent.npz` holds the RynnLAM v2 labels (`ktoken_zcam`, 608-dim); `data/manifest/action_stats.json` is the quantile statistics training normalizes with. `data/manifest/chunks_00000.jsonl` stores `src`/`lat` **relative to `data/`**, which `rynnworld_latent.manifest_dataset` resolves at load time — that is what keeps the bundle portable across machines. Videos and latents are unmodified copies, so the frames the VAE sees match the frames the latents were labelled from. Full training/evaluation corpora are **not** redistributed — each has its own access terms (see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)).

---

## Data format (manifest)

RynnWorld-Latent trains from a **manifest** directory of video chunks paired with latent actions. Each action is a 608-dim RynnLAM `ktoken_zcam` latent `[k_tokens 512 | z 64 | camera 32]`, labeled at gap=4 / pair_stride=4 — already at action rate (one latent per 4 RGB frames = one Wan-VAE latent frame). A chunk is therefore **81 pixel frames** (1 conditioning + 80 future) paired with **20 consecutive latents**. The bundled `data/manifest/` is a working example:

```
<manifest>/chunks_*.jsonl     # one record per chunk: ds/ep/view/src/vid/lat/start/nf/na/cap/fps/as/lat_start/lat_stride
<manifest>/action_stats.json  # 608-dim quantile stats (q01/q99/mean/std); training normalizes to [-1,1] with these
<manifest>/summary.json       # chunk count, datasets, protocol ("v2 ktoken_zcam 608")
```

`src`/`lat` may be relative to the manifest's parent (the bundled data) or absolute. Each latent `.npz` (produced by RynnLAM labeling) carries `latent_action [T,608] float16`, `pair_indices [T,2] int32`, and a `meta` JSON with the protocol (`start_frame`/`gap`/`pair_stride`/checkpoint sha). `rynnworld_latent/manifest_dataset.py::RynnWorldManifestDataset` reads pixels straight from the source video at a frame offset (`start_frame + start`) — nothing is materialized — and `sft_dataset.py` wraps it in cosmos's `ActionTransformPipeline` with a resumable streaming shuffle (seed keyed on the committed trainer iteration, so a restart continues the data order).

The dataset **fails fast at init** if `action_stats.json` is missing or not 608-dim (silent degradation would train on raw latents with a loss that still looks finite). The intentional unnormalized path is `action_normalization=None`. Plain `.mp4`/`.webm` sources decode via PyAV; `rovidx_tar://` / zarr / hdf5 sources go through RynnLAM's `rynnlam.video.FrameReader` (see `rynnworld_latent/rynnlam_bridge.py`). `STAGED_ROOT=""` (the default) reads source videos in place; `RYNNWORLD_SRC_REMAP='old=>new'` rewrites source paths if a mount moved.

Latent actions themselves are produced by the bundled RynnLAM labeler — see [`rynnlam/README.md`](rynnlam/README.md).

---

## Training & resume (single node)

`scripts/train.sh` is the single-node launcher; `scripts/train_8gpu.sh` is the submit wrapper (bundled-data quickstart by default). **You must go through `scripts/train.py`** (it registers the experiments + applies the compat/conditioning patches); launching `-m cosmos_framework.scripts.train` directly does neither.

```
scripts/train.sh ─ torchrun ─ scripts/train.py
   ├─ registers the rynnworld_latent_edge_manifest* experiments into Hydra's ConfigStore
   ├─ compat patches: av.option / lerobot stub / local tokenizer / fully_shard / _init_weights /
   │                  sched_setaffinity / SDPA varlen attention / DCP save planner
   ├─ action zero-init patch
   ├─ register_action_conditioning_patches() + register_cond_force_patches()   # film
   └─ runpy hand-off to cosmos_framework.scripts.train
```

### Recipes (`rynnworld_latent/experiment_config.py`)

| Experiment | Trains | Use |
|---|---|---|
| `rynnworld_latent_edge_manifest` | generation branch + action heads (the `optimizer.keys_to_select` allowlist) | cheaper runs, single-GPU smoke |
| `rynnworld_latent_edge_manifest_fullft` | **everything** (`keys_to_select=[]`), fps pinned to 10.0 | full-parameter fine-tune from the Cosmos3-Edge base |
| `rynnworld_latent_edge_manifest_v3_fullft` | same as `_fullft` but `fps=None` (per-record real fps) | **the released weights' recipe** — reproduces the v3 checkpoint |
| `rynnworld_latent_edge_posttrain` | everything, warm-started from a *trained film* checkpoint (only `net_ema.` skipped) | downstream adaptation to your own embodiment |

### Full film recipe

```bash
export MANIFEST_DIR=/path/to/manifest          # or omit to use the bundled data/ via train_8gpu.sh
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-dcp
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=./outputs

export RYNNWORLD_TEXT_FREE=1 \
       RYNNWORLD_ACTION_CFG_DROPOUT=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1 \
       RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1 RYNNWORLD_COND_FORCE=0.3 \
       RYNNWORLD_COND_FORCE_FLOOR=0.75 RYNNWORLD_ACTION_FRAME_INJECT=1 \
       RYNNWORLD_ACTION_TOWER_SPLIT=1 RYNNWORLD_ACTION_FILM=1

NGPU=8 TOML=configs/train/edge_fullft.toml bash scripts/train.sh
```

Run names land under `$IMAGINAIRE_OUTPUT_ROOT/rynnworld_latent/rynnworld_latent_sft/<name>/checkpoints/`.

### Training strategy (the released recipe)

- **Full-parameter fine-tune** — `keys_to_select=[]`, all 3.37B parameters train.
- **5x LR on the action heads** — `lr_multipliers` for `action2llm`/`llm2action`/`action_modality_embed` = 5.0; base `lr=2e-4`, `weight_decay=0.05`, `betas=[0.9,0.99]`.
- **Action pathway zero-initialised** — Cosmos3-Edge ships a *trained* 64-dim action pathway (32 embodiment domains; domain 3 = `hand_pose`, hence `DOMAIN_ID=3`), but those rows have **no correspondence** in `[k_tokens 512 | z 64 | camera 32]`, so `scripts/train.py::_zero_action_pathway` zeroes `action2llm` + `action_modality_embed` and they are learned from scratch. Action layers are in `keys_to_skip_loading`, so the zero-init survives the DCP main load.
- **LR schedule** — `LambdaLinear`, `f_max=0.4`, `warm_up_steps=[500]`; in the TOML `cycle_lengths` **must equal** `trainer.max_iter` or the anneal never completes (released recipe: 20000 for both, `f_min=[0.04]`).
- **DCP warm-start** — `load_path=$BASE_CHECKPOINT_PATH`, `load_training_state=False`, `strict_resume=False`, `keys_to_skip_loading` covers the action layers and `net_ema.`.
- **Batch / memory** — `max_samples_per_batch=4` for full-FT (full-FT triples optimizer state; on the SDPA fallback without flash_attn, 8 would OOM). Optimizer: `FusedAdam`; without transformer_engine override `optimizer.optimizer_type=adamw optimizer.fused=true`. bf16 + FSDP shard-8.
- **forward_dynamics semantics** — only frame 0 is a visual condition and **all 20 actions are conditions** (none predicted), so `llm2action` runs a dummy path with no effect on training; only `action2llm` (encoding) + the injection paths matter.

### Resume

Resume continues from a saved checkpoint under `.../checkpoints/iter_XXXXXXX`: point `checkpoint.load_path` at that directory and set `checkpoint.load_training_state=true` to restore the optimizer/scheduler/iteration (the resumable shuffle seed is keyed on the committed iteration, so the data order continues correctly). Export the **same film switches** used in training, or the model structure will mismatch.

Validate the data path without a GPU:

```bash
python scripts/inference/rollout.py --dry-run --manifest-dir "$MANIFEST_DIR" --indices 0,1
```

### Downstream embodiment post-training (your own data)

The released model is a general film checkpoint; the highest-value use is **adapting it to your own embodiment**. `rynnworld_latent_edge_posttrain` warm-starts from a *trained* RynnWorld-Latent film checkpoint (not the Cosmos3-Edge base) and fine-tunes on your manifest. Because that checkpoint already has trained action heads, this recipe narrows `keys_to_skip_loading` to `["net_ema."]` only, so the film action pathway is **preserved** instead of reseeded (the from-base recipes drop `action2llm`/`llm2action`/`action_modality_embed` to zero-init them). `load_training_state` stays False — weights carry over; optimizer/scheduler/iteration reset. It is data-agnostic: the embodiment is selected purely by `MANIFEST_DIR`, and nothing in the recipe names a dataset.

Turnkey flow — download the released weights, re-serialize to DCP once, then post-train:

```bash
# 1. Download the released film checkpoint (safetensors) — same weights the rollout section uses.
python -c "from huggingface_hub import snapshot_download; snapshot_download('Alibaba-DAMO-Academy/RynnWorld-Latent', local_dir='./weights/RynnWorld-Latent')"

# 2. One-time CPU-only re-serialization to DCP (the trainer warm-starts from DCP, not safetensors).
#    No GPU, no model instantiation, no VAE. Add --skip-ema to drop net_ema.* and halve the output.
python scripts/checkpoints/convert_released_to_dcp.py \
    --safetensors ./weights/RynnWorld-Latent \
    --out         ./weights/RynnWorld-Latent-dcp

# 3. Post-train on your own embodiment manifest (or data/manifest for a pipeline smoke).
export MANIFEST_DIR=/path/to/your/embodiment/manifest   # documented format above; RynnLAM-labelled latents
export BASE_CHECKPOINT_PATH=./weights/RynnWorld-Latent-dcp   # the DCP from step 2
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=./outputs
# export the SAME RYNNWORLD_* film switches the released checkpoint was trained with (above)

NGPU=8 TOML=configs/posttrain/downstream.toml bash scripts/train.sh
```

`configs/posttrain/downstream.toml` is a template (lr `5e-5` ≈ 4× below the from-base recipe, 3000 iters — retune to your corpus). `BASE_CHECKPOINT_PATH` must be a **DCP** checkpoint (the cosmos trainer warm-starts from DCP, not safetensors): step 2's `convert_released_to_dcp.py` produces one from the released weights without any training, or point it at a stage-1 `edge_fullft` / `v3_fullft` output you trained yourself. On a stock `requirements.txt` environment (no NVIDIA `transformer_engine`/`apex`), `scripts/train.py` auto-injects a fused-AdamW optimizer (`optimizer.optimizer_type=adamw optimizer.fused=true`), so no manual override is needed; pin `optimizer.optimizer_type=...` yourself to keep a different optimizer.

> **`fps=None` (v3) caveat.** `rynnworld_latent_edge_manifest_v3_fullft` sets the dataset `fps=None`, so `manifest_dataset.py` uses **each record's own probed fps** instead of overriding everything to 10.0. That fps is the mRoPE temporal-step denominator (`base_fps/fps = 24/fps`), so it globally rescales the time-position encoding: a checkpoint trained at fps=10.0 **cannot** be warm-started into v3, or vice-versa. **Your `chunks_*.jsonl` must carry the real per-record `fps`** — a wrong/placeholder value silently misaligns the mRoPE timestep (finite loss, no error); a missing one raises at load. Pin fps=10.0 (`_fullft`) only if your corpus is genuinely uniform 10 fps.

### Bundled per-embodiment quickstart (real sample, runs out of the box)

Each downstream embodiment ships with **one real teleop sample** under `data/<embodiment>/` plus a launcher, so you can exercise the full per-embodiment post-train path immediately. A launcher warm-starts from the released film DCP (step 2 above), points `MANIFEST_DIR` at that embodiment's bundled sample, and exports the film switches — the per-embodiment launchers differ only in which sample they load.

| Embodiment | Launcher | Bundled sample |
|---|---|---|
| **Astribot-S1** | `scripts/posttrain/astribot_s1.sh` | `data/astribot_s1/` — 1 record, 81 frames / 20 latents, 30 fps |

```bash
export BASE_CHECKPOINT_PATH=./weights/RynnWorld-Latent-dcp   # convert_released_to_dcp.py output (step 2)
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth

NGPU=8 bash scripts/posttrain/astribot_s1.sh              # full 3000-iter run
NGPU=1 MAX_ITER=3 bash scripts/posttrain/astribot_s1.sh   # single-GPU smoke
```

`MAX_ITER` caps both `trainer.max_iter` and the scheduler `cycle_lengths` (they must match). To post-train on your **own** corpus instead of the bundled sample, override the manifest: `MANIFEST_DIR=/path/to/yours bash scripts/posttrain/astribot_s1.sh`.

> **What ships / what doesn't.** The bundled video is an **unmodified** copy (so the frames the VAE sees match the frames the actions were labelled from); internal absolute paths, corpus sample counts, and manifest fingerprints are stripped from the shipped `chunks_*.jsonl` / `action_stats.json` / latent `meta`, and `src`/`lat` are relative to `data/<embodiment>/` exactly like the generic `data/manifest` bundle. The sample's 608-dim latents are **pre-generated** and included; this release bundles RynnLAM *video* labeling (`rynnlam/`) for the generic latent route, but not the embodiment-specific physical-action→MLP labeling pipeline that produced these particular latents — treat the sample as a runnable reference and label your own data with RynnLAM.

---

## Inference & evaluation

### Released film weights (download → roll out, no training)

The fine-tuned **RynnWorld-Latent film checkpoint** (608-dim two-tower action conditioning + frame FiLM) is published publicly. Download it and roll out directly — no training and no DCP conversion:

```bash
# Hugging Face (China mirror: export HF_ENDPOINT=https://hf-mirror.com)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Alibaba-DAMO-Academy/RynnWorld-Latent', local_dir='./weights/RynnWorld-Latent')"
# or ModelScope (pip install modelscope)
python -c "from modelscope import snapshot_download; snapshot_download('DAMO_Academy/RynnWorld-Latent', local_dir='./weights/RynnWorld-Latent')"
```

The release ships safetensors in the native Cosmos3 VFM layout — `net.*` (regular) plus `net_ema.*` (EMA) mirrors — and `rollout.py` loads the `net.*` weights straight from the directory. It still needs the gated **Wan2.2 VAE** (from `bash scripts/setup/download_weights.sh`, above → `WAN_VAE_PATH`) and the **film switches** the checkpoint was trained with:

```bash
export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export RYNNWORLD_TEXT_FREE=1 \
       RYNNWORLD_ACTION_TOWER_SPLIT=1 RYNNWORLD_ACTION_FRAME_INJECT=1 RYNNWORLD_ACTION_FILM=1 \
       RYNNWORLD_ACTION_CFG_DROPOUT=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1
python scripts/inference/rollout.py \
  --manifest-dir data/manifest \
  --checkpoint ./weights/RynnWorld-Latent \
  --indices 0,1,2 --num-steps 35 --guidance 1.5 --no-ema --out /tmp/rollout
```

`--no-ema` is required here (see the note above): the safetensors loader consumes `net.*` and ignores `net_ema.*`. On the three bundled sample chunks this reproduces **PSNR(gen,gt) ≈ 21–23 dB**, above the 15–22 dB static-first-frame baseline, with motion(gen)/motion(gt) ≈ 1.0–1.3 — i.e. the model predicts real action-conditioned motion rather than copying frame 0.

### Your own trained checkpoint (DCP)

```bash
export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
python scripts/inference/rollout.py \
  --manifest-dir "$MANIFEST_DIR" \
  --checkpoint "$IMAGINAIRE_OUTPUT_ROOT/.../checkpoints/iter_000006000" \
  --indices 0,1,2,3 --num-steps 35 --guidance 1.5 --out /tmp/rollout
```

It writes `gen_*.mp4` / `gt_*.mp4` and prints per-sample PSNR(gen,gt), the static-first-frame baseline, and the motion ratio. **Export the same film switches used in training.** For a DCP checkpoint, EMA loads by default; pass `--no-ema` for a run that did not save an EMA.

---

## Examples / demos

Two runnable demos showcase what action-conditioning buys. Both load a checkpoint exactly like `rollout.py` (same env, same `RYNNWORLD_ACTION_*` film switches) and need `ffmpeg` on `PATH`.

**Cross-action transfer** (`examples/cross_action_transfer.py`) — the core evidence that the model follows the *action*, not just the first frame. For a pair of chunks (A, B) it renders A's first frame + A's own action (`gen_self`) and A's first frame + **B's** latent action (`gen_cross`); if conditioning works, `gen_cross` moves like B while keeping A's appearance. Writes side-by-side `A_gt | B_gt | gen_cross` panels.

```bash
export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
python examples/cross_action_transfer.py \
  --manifest-dir "$MANIFEST_DIR" --checkpoint /path/to/iter_000008000 \
  --pairs 0:1,2:3 --out outputs/xfer        # add --staged-root if you staged 480p copies
```

**Inpainted first frame** (`scripts/inference/inpaint_rollout.py`) — the data-synthesis mode: replace frame 0 with your own edited image (e.g. an inpainted or embodiment-swapped scene) but keep the source chunk's 20 latent actions, so the generated motion follows the original while the appearance follows your edit. Writes the generation and its GT counterpart.

```bash
python scripts/inference/inpaint_rollout.py \
  --manifest-dir "$MANIFEST_DIR" --checkpoint /path/to/iter_000008000 \
  --index 0 --cond-image /path/to/edited_frame0.png --out outputs/inpaint_rollout.mp4
```

---

## RynnLAM (bundled latent-action model)

[`rynnlam/`](rynnlam) is the bundled latent-action model (the LAM series work) that produces the conditioning signals. It is a self-contained project with its own `pyproject.toml`, `LICENSE` (Apache-2.0) and `NOTICE`. **This bundle ships RynnLAM's inference, dense video labeling, and LARYBench-compatible evaluation paths only** — RynnLAM *training* is not included. RynnWorld-Latent imports `rynnlam.video` (FrameReader for `rovidx_tar://` / zarr / hdf5 sources) and the encoder at runtime; the bridge is `rynnworld_latent/rynnlam_bridge.py` (set `RYNNLAM_ROOT` to override the bundled copy).

Label videos and run the latent-action benchmark per [`rynnlam/README.md`](rynnlam/README.md). RynnLAM reads `.mp4`/`.webm` (av), `.hdf5` (RoboMIND), zarr directories (EgoVerse) and `rovidx_tar://` (RoVid-X); any source resolution is normalized by aspect-ratio bucket (square 280×280, 4x3 238×322, 16x9 210×364) to a patch-14 multiple before encoding.

---

## Project layout

```
RynnWorld-Latent/
  rynnworld_latent/   # the world-model package (action conditioning, datasets,
                      #   experiment configs, inference helpers)
  scripts/
    setup/            #   download_weights.sh (gated weights + DCP conversion)
    train.py/.sh      #   training entrypoint + single-node torchrun launcher
    train_8gpu.sh     #   single-node submit; bundled-data quickstart by default
    inference/        #   rollout.py (roll out vs GT), inpaint_rollout.py (custom first frame)
    posttrain/        #   astribot_s1.sh (per-embodiment downstream post-train launcher)
    checkpoints/      #   convert_edge_to_dcp.py (base Edge HF -> DCP),
                      #   convert_released_to_dcp.py (released film safetensors -> DCP)
    test_world_model.sh   # staged data / lam / train / infer checks on data/
    test_lam_inference.py # re-encode data/videos with RynnLAM, diff vs data/latents
  examples/           # cross_action_transfer.py (+ _video_utils.py) demo
  configs/            # recipe TOMLs (train/ full-FT, posttrain/ downstream, examples/ smoke)
  data/               # bundled samples: manifest/ (3 generic chunks) + <embodiment>/ (1 real
                      #   teleop sample each, e.g. astribot_s1/) for the post-train quickstarts
  rynnlam/            # bundled latent-action model (own LICENSE/NOTICE/pyproject)
  third_party/
    cosmos-framework/ # NVIDIA training stack, CODE ONLY (no weights)
    cosmos_tokenizers/# text tokenizer + model cards (no weights)
```

### Key files

| File | Role |
|---|---|
| `rynnworld_latent/action_conditioning.py` | **all of film**: two-tower encoder, frame-aligned injection, per-frame FiLM, action CFG dropout, inference guidance rewrite, condition forcing |
| `rynnworld_latent/experiment_config.py` | Hydra recipes, text-free switch, optimizer/scheduler strategy |
| `rynnworld_latent/dataset.py` | shared video decoding (PyAV / FrameReader), resolution-tier bucketing, action normalization; `DOMAIN_ID`, `ACTION_DIM` |
| `rynnworld_latent/manifest_dataset.py` | the manifest dataset: frame-offset reads from source or staged video, `RYNNWORLD_SRC_REMAP`, per-sample retry on bad media |
| `rynnworld_latent/sft_dataset.py` | SFT factory: wraps the manifest dataset in `ActionTransformPipeline`, resumable shuffle |
| `rynnworld_latent/rynnlam_bridge.py` | lazy bridge to the bundled RynnLAM: `FrameReader` for special sources, `start_frame` from npz meta |
| `rynnworld_latent/inference.py` | shared model loading (`load_world_model`), rollout batch construction, local-tokenizer patch, PSNR / temporal-diff metrics |
| `scripts/train.py` / `train.sh` / `train_8gpu.sh` | training entrypoint (registration + patches) / single-node torchrun launcher / submit wrapper |
| `scripts/posttrain/astribot_s1.sh` | per-embodiment downstream post-train launcher (bundled real sample; override `MANIFEST_DIR` for your own) |
| `scripts/checkpoints/convert_edge_to_dcp.py` | base Cosmos3-Edge HF safetensors → DCP (builds the model; for stage-1 from-base training) |
| `scripts/checkpoints/convert_released_to_dcp.py` | released film safetensors (`net.*`) → DCP, CPU-only & model-free (warm-start for downstream post-training) |
| `scripts/setup/download_weights.sh` | gated Cosmos3-Edge download + DCP conversion |
| `scripts/inference/rollout.py` | first-frame + latent-action rollout (`gen_*.mp4` / `gt_*.mp4` + PSNR) |
| `scripts/inference/inpaint_rollout.py` | rollout with a custom (e.g. inpainted) first frame, reusing the source latent actions |
| `examples/cross_action_transfer.py` | cross-action transfer demo (A's frame + B's action) with side-by-side panels |
| `scripts/test_world_model.sh` / `scripts/test_lam_inference.py` | staged end-to-end checks on the bundled data (world model / RynnLAM) |
| `configs/train/edge_fullft.toml` | **the released full-parameter recipe** |
| `configs/posttrain/downstream.toml` | data-agnostic downstream post-training template (warm-start from a released/stage-1 DCP) |
| `configs/examples/edge_manifest_local.toml` | single-GPU smoke recipe |

---

## Contributing

Thanks for your interest in contributing!

1. Fork the repo and branch from `main`. Install editable: `pip install -e ".[video-sources]"`, then `export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"`. Download the gated weights (`bash scripts/setup/download_weights.sh`) before anything that loads a model.
2. **Smoke check** (no GPU): `python scripts/inference/rollout.py --dry-run --manifest-dir <manifest> --indices 0,1`.
3. Keep action-conditioning behaviour behind its `RYNNWORLD_*` env switches (`rynnworld_latent/action_conditioning.py`); new switches should default to the current behaviour so existing runs are unchanged.
4. One logical change per PR with a clear *why*. Update this README when you change behaviour or flags. New dependencies go in `pyproject.toml` **and** 