<div align="center">

## RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions


</div>

<p align="center">
  💫 <a href="https://alibaba-damo-academy.github.io/RynnWorld-Latent.github.io"><b>Project Page</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent"><b>Hugging Face</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  🤖 <a href="https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-Latent"><b>ModelScope</b></a> &nbsp;&nbsp;|&nbsp;&nbsp;
  📄 <a href="https://arxiv.org/abs/2610.36791"><b>arXiv</b></a>
</p>

---

## 🌟 Abstract

Modeling world dynamics for dexterous robotics is hard: embodiment interfaces are heterogeneous, robot data coverage is limited, and fine-grained action labels are scarce. RynnWorld-Latent is a cross-embodiment video world model built for both future prediction and scalable robot data generation. It addresses the action-label bottleneck by using **continuous latent actions** — a compact code describing *what changed* between two observations — so human egocentric video and robot data are modeled jointly without embodiment-specific control commands. Built on a pretrained video diffusion transformer (NVIDIA **Cosmos3-Edge**, 3.37B), it is **text-free** and adds **explicit frame-aligned action conditioning**: given a first-frame observation + 20 latent actions, predict the next 80 video frames. It is trained on a large cross-embodiment mixture (32 datasets, >265M frames) and post-trained on small-scale target-robot data to align the latent-action space with the target robot's control interface.

Beyond prediction, the model is a **data-generation engine** in two modes:

- **Interactive digital teleoperation** — the post-trained model is conditioned on target-robot actions, so operator control signals drive video generation.
- **Data synthesis** — given a source video, edit its first frame to swap the embodiment or environment, reuse the source video's latent actions, and roll out a new video whose interaction dynamics follow the source but appear under the new embodiment/environment.


## 📖 Model weights

| Artifact | What it is | Where |
| --- | --- | --- |
| **RynnWorld-Latent** | fine-tuned film checkpoint (608-dim two-tower action conditioning + frame FiLM), safetensors `net.*` / `net_ema.*` | [Alibaba-DAMO-Academy/RynnWorld-Latent](https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Latent) · [DAMO_Academy/RynnWorld-Latent](https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-Latent) |
| **NVIDIA Cosmos3-Edge** | pretrained base (3.37B) + Wan2.2 VAE — **gated**, requested from NVIDIA | `bash scripts/setup/download_weights.sh` after `huggingface-cli login` |

This repository redistributes **code plus bundled sample chunks only — no weights**; weight licenses are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The download script also converts the base to the DCP layout cosmos trains on and prints the `BASE_CHECKPOINT_PATH` / `WAN_VAE_PATH` exports you need. The released RynnWorld-Latent checkpoint rolls out directly with no conversion (see [Inference and evaluation](#-inference-and-evaluation)); downstream post-training wants it as DCP (`scripts/checkpoints/convert_released_to_dcp.py`, CPU-only).

## 🌳 Repository layout

```
RynnWorld-Latent/
├── rynnworld_latent/   # world-model package (action conditioning, datasets, recipes, inference)
├── scripts/            # train / test entrypoints; inference/, posttrain/, checkpoints/, setup/
├── examples/           # cross_action_transfer.py demo
├── configs/            # recipe TOMLs: train/ full-FT, posttrain/ downstream, examples/ smoke
├── docs/               # architecture & training deep references
├── data/               # bundled samples: manifest/ (3 chunks) + astribot_s1/, marvin_wuji/
├── rynnlam/            # bundled latent-action model (own LICENSE/NOTICE/pyproject)
├── third_party/        # vendored cosmos-framework + cosmos_tokenizers (CODE ONLY, no weights)
├── requirements.txt    # pinned reference environment
├── NOTICE              # third-party attribution
└── README.md           # this file
```

## 📦 Installation

Linux, Python ≥ 3.10, CUDA GPU.

```bash
git clone https://github.com/alibaba-damo-academy/RynnWorld-Latent && cd RynnWorld-Latent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[video-sources]"                     # or: pip install -r requirements.txt

# cosmos-framework is vendored, not on PyPI — put it (and the repo) on PYTHONPATH:
export PYTHONPATH="$PWD:$PWD/third_party/cosmos-framework"
```

Pins matter (`transformers>=4.57,<5`, `torch>=2.7,<2.11`, `wandb<0.28.2`, `numpy<2`; rationale in [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md#dependency-pins)). NVIDIA-image-only pieces (`apex`, `flash_attn`, `transformer_engine`, …) are **not** required — `scripts/train.py` patches around them (SDPA varlen fallback, inert lerobot stub).

## 🚀 Quickstart

`data/` ships three real sample chunks (videos + RynnLAM latents + a v2 manifest) so a fresh clone trains and evaluates out of the box. Download the gated base weights first (see [Model weights](#-model-weights)), then:

```bash
bash scripts/train_8gpu.sh          # quickstart: NGPU=1, bundled data/, 3 iters
```

With no `MANIFEST_DIR` set it trains on the bundled `data/manifest` using the single-GPU smoke recipe; override with `QUICKSTART_NGPU` / `QUICKSTART_ITERS`, or set `MANIFEST_DIR` to train on your own data. The quickstart writes its end-of-run checkpoint to local disk (`$TMPDIR/rynnworld_latent_quickstart`) by default — set `IMAGINAIRE_OUTPUT_ROOT` explicitly to keep the artifact.

Verify the install stage by stage (each independently runnable; CPU-only unless noted):

```bash
bash scripts/test_world_model.sh --stage data          # manifest + latents: shapes, normalization, domain id
bash scripts/test_world_model.sh --stage lam           # re-encode videos with RynnLAM, diff vs shipped latents (GPU; needs --lam-ckpt / RYNNLAM_CKPT)
bash scripts/test_world_model.sh --stage train-smoke   # 3 iters, proves the patches + DCP load (GPU)
bash scripts/test_world_model.sh --stage train         # 50 iters, loss should move (GPU)
bash scripts/test_world_model.sh --stage infer --checkpoint /path/to/iter_NNNNNNN  # rollout vs GT (GPU)
bash scripts/test_world_model.sh --stage all
```

It exports the full film switch set (override with `--no-film`) and writes to `$OUT_ROOT`, never into the repo.

## 🗄️ Data format (manifest)

Training reads a **manifest** directory of video chunks paired with latent actions. Each action is a 608-dim RynnLAM `ktoken_zcam` latent `[k_tokens 512 | z 64 | camera 32]` at gap=4 / pair_stride=4 — one latent per 4 RGB frames = one Wan-VAE latent frame — so a chunk is **81 pixel frames** (1 conditioning + 80 future) paired with **20 consecutive latents**. The bundled `data/manifest/` is a working example:

```
<manifest>/chunks_*.jsonl     # one record per chunk: ds/ep/view/src/vid/lat/start/nf/na/cap/fps/as/lat_start/lat_stride
<manifest>/action_stats.json  # 608-dim quantile stats (q01/q99/mean/std); training normalizes to [-1,1] with these
<manifest>/summary.json       # chunk count, datasets, protocol ("v2 ktoken_zcam 608")
```

`src`/`lat` are relative to the manifest's parent (or absolute) — which keeps the bundle portable. Pixels are read straight from the source video at a frame offset; nothing is materialized. The dataset **fails fast** if `action_stats.json` is missing or not 608-dim. `.mp4`/`.webm` decode via PyAV; `rovidx_tar://` / zarr / hdf5 go through RynnLAM's `FrameReader`. Label your own videos with the bundled RynnLAM (see [`rynnlam/README.md`](rynnlam/README.md)).

## ⚙️ Training

`scripts/train.sh` is the single-node launcher and `scripts/train_8gpu.sh` the submit wrapper. **Always go through `scripts/train.py`** — it registers the experiments and applies the compat/conditioning patches; launching cosmos directly does neither.

| Experiment | Trains | Use |
|---|---|---|
| `rynnworld_latent_edge_manifest` | generation branch + action heads | cheaper runs, single-GPU smoke |
| `rynnworld_latent_edge_manifest_fullft` | everything, fps pinned to 10.0 | full-parameter fine-tune from the Cosmos3-Edge base |
| `rynnworld_latent_edge_manifest_v3_fullft` | same, `fps=None` (per-record real fps) | **the released weights' recipe** |
| `rynnworld_latent_edge_posttrain` | everything, warm-started from a trained film checkpoint | downstream adaptation to your own embodiment |

Full film recipe:

```bash
export MANIFEST_DIR=/path/to/manifest
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

Checkpoints land under `$IMAGINAIRE_OUTPUT_ROOT/rynnworld_latent/rynnworld_latent_sft/<name>/checkpoints/`.

**Resume:** point `checkpoint.load_path` at `.../checkpoints/iter_XXXXXXX` and set `checkpoint.load_training_state=true`. **Export the same film switches as the original run**, or the model structure mismatches (`frame_gate` / `frame_film` missing).

**Post-train on your own embodiment:** download the released weights, re-serialize to DCP once (CPU-only), then run `configs/posttrain/downstream.toml` against your manifest:

```bash
python scripts/checkpoints/convert_released_to_dcp.py \
    --safetensors ./weights/RynnWorld-Latent --out ./weights/RynnWorld-Latent-dcp
export BASE_CHECKPOINT_PATH=./weights/RynnWorld-Latent-dcp   # + MANIFEST_DIR / WAN_VAE_PATH / film switches
NGPU=8 TOML=configs/posttrain/downstream.toml bash scripts/train.sh
```

Per-embodiment turnkey launchers with bundled real samples: `scripts/posttrain/astribot_s1.sh`, `scripts/posttrain/marvin_wuji.sh` (`NGPU=1 MAX_ITER=3 bash …` for a single-GPU smoke). Full post-training details, strategy notes and the `fps=None` caveat: [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).

Validate the data path without a GPU:

```bash
python scripts/inference/rollout.py --dry-run --manifest-dir "$MANIFEST_DIR" --indices 0,1
```

## 🧠 Latent-Action Model (RynnLAM)

[`rynnlam/`](rynnlam) produces the conditioning signals: a self-contained project with its own `pyproject.toml`, Apache-2.0 `LICENSE` and `NOTICE`. **This bundle ships inference, dense video labeling and LARYBench-compatible evaluation only — RynnLAM training is not included.** RynnWorld-Latent imports `rynnlam.video` (FrameReader) and the encoder at runtime via `rynnworld_latent/rynnlam_bridge.py` (`RYNNLAM_ROOT` overrides the bundled copy). Sources: `.mp4`/`.webm`, `.hdf5` (RoboMIND), zarr (EgoVerse), `rovidx_tar://` (RoVid-X); resolutions are normalized by aspect-ratio bucket before encoding. Usage: [`rynnlam/README.md`](rynnlam/README.md).

## 🚀 Inference and Evaluation

The released **film checkpoint** rolls out directly, no training and no DCP conversion:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Alibaba-DAMO-Academy/RynnWorld-Latent', local_dir='./weights/RynnWorld-Latent')"
# or ModelScope: python -c "from modelscope import snapshot_download; snapshot_download('DAMO_Academy/RynnWorld-Latent', local_dir='./weights/RynnWorld-Latent')"

export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export RYNNWORLD_TEXT_FREE=1 \
       RYNNWORLD_ACTION_TOWER_SPLIT=1 RYNNWORLD_ACTION_FRAME_INJECT=1 RYNNWORLD_ACTION_FILM=1 \
       RYNNWORLD_ACTION_CFG_DROPOUT=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1

python scripts/inference/rollout.py \
  --manifest-dir data/manifest \
  --checkpoint ./weights/RynnWorld-Latent \
  --indices 0,1,2 --num-steps 35 --guidance 1.5 --no-ema --out /tmp/rollout
```

- `--no-ema` is **required** for the released safetensors: the direct loader reads `net.*` and ignores the `net_ema.*` mirrors. For your own **DCP** checkpoints EMA loads by default (pass `--no-ema` if the run saved none).
- Evaluating any checkpoint requires the **same film switches** it was trained with.
- On the three bundled chunks this reproduces **PSNR(gen,gt) ≈ 21–23 dB** (fps=None, v3 semantics), above the 15–22 dB static-first-frame baseline, motion ratio ≈ 1.0–1.2.

`rollout.py` writes `gen_*.mp4` / `gt_*.mp4` plus per-sample PSNR, the static baseline and the motion ratio. Point `--checkpoint` at your own `.../checkpoints/iter_XXXXXXX` for a trained DCP. Rollout-vs-GT on a manifest is a dynamics-consistency check, **not** a task score — no downstream benchmark orchestration ships here.

## 📑 Citation

If you find this project useful, please cite:

```bibtex
@article{rynnworldlatent,
  title  = {RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions},
  author = {Haoyu Zhao and Yinzhou Tang and Zixuan Wang and Minghao Zhu and Dongchi Huang and Xingyue Zhao and Qin Zhao and Kehan Li and Siteng Huang and Xin Li and Zhenyu Zeng and Mingxiu Chen and Deli Zhao and Zhongyu Li},
  journal= {arXiv preprint arXiv:2610.36791},
  year   = {2026},
}
```

## ⚖️ License & Attribution

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). This repository vendors the NVIDIA cosmos-framework (code only) and cosmos tokenizers, and bundles RynnLAM; third-party components and the gated weight licenses are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model weights, the pretrained bases and the full training/evaluation corpora are separate resources and are not granted by this source license — obtain each under its own access terms.
