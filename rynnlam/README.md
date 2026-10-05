# RynnLAM

RynnLAM learns video-based latent action representations for vision-language-action models. **This bundled copy ships the inference, dense video labeling, and LARYBench-compatible evaluation paths only** — the DA3-Large/K-token encoder, the `RynnLAMEncoder` inference API, the `label_latent_dense.py` labeling CLI, and the `evaluate.py` benchmark interface. RynnLAM *training* (the trainer, losses, and data-preparation tools) is intentionally not part of this bundle: RynnWorld-Latent consumes RynnLAM's labeled latent actions and its `rynnlam.video` frame reader, and only needs to run the encoder.

This is a source release. Datasets, model checkpoints, private infrastructure, experiment logs, and external benchmark implementations are not included.

## Method and representations

Two RGB frames are encoded jointly by DA3-Large. A learnable-query compressor reads the target-frame features and returns K tokens. A reconstruction decoder uses the source-frame features and these tokens to predict target features (trained upstream with a motion-weighted, per-patch normalized residual loss). This bundle runs the encoder for inference and labeling; it does not include the training graph.

The feature-source path is **not** the 64-dimensional global z and is **not** the 128-dimensional motion-hint field. The actual representation dimensions are:

| Representation | Shape per frame pair |
| --- | --- |
| K tokens, `k_token_source: features` | `[K, 2048]` |
| K tokens, `k_token_source: hints` | `[K, motion_hint_dim]` |
| Global z | `[latent_dim]`, reference value 64 |
| Pooled motion hints | `[motion_hint_dim]`, reference value 128 |

For the supplied feature-source recipes, K=2/4/8 therefore gives **4096/8192/16384 flattened values**, not 256/512/1024. Loading is strict: output dimensions are not silently changed to fit a checkpoint.

DA3 uses cross-frame attention, so its source-frame features can already depend on both input frames. A positive zero-token ablation gain measures decoder dependence on tokens, not a proof of action alignment. The normalized residual objective likewise does not mathematically eliminate all shortcuts. Evaluate representations with a fixed protocol; do not select a model from training loss alone.

## Installation

Python 3.10 or newer is required. Install a PyTorch build compatible with your GPU/CUDA runtime, then from this directory:

```bash
python -m pip install -e '.[video,test]'
```

Core model use does not require LARYBench. PyAV (`av`) is used for video labeling; to read the non-file sources `rynnlam.video.FrameReader` supports (zarr/EgoVerse, hdf5/RoboMIND), install the matching readers via the top-level RynnWorld-Latent `video-sources` extra.

The backbone can be initialized from a local DA3-Large `model.safetensors`, or a compatible Hugging Face repository identifier. A complete RynnLAM checkpoint already includes its backbone and does not require the old backbone file location. Acquire weights under their own license; this repository does not redistribute them.

## Encode and label videos

```python
import torch
from rynnlam.inference import RynnLAMEncoder

encoder = RynnLAMEncoder("checkpoint.pt", device="cuda", normalize=True)
images = torch.rand(1, 2, 210, 364, 3)  # RGB floats in [0, 1]
tokens = encoder(images, representation="ktoken")  # [1, K, 2048]
```

For stride-four motion labels from one video (the labeling CLI default):

```bash
python label_latent_dense.py --video /path/to/video.mp4 \
  --checkpoint /path/to/checkpoint.pt --output-dir ./outputs/labels_stride4 \
  --representation ktoken --gap 4 --pair-stride 4
```

Row i encodes `(frame 4*i, frame 4*i+4)`: `(0,4), (4,8), (8,12), ...`. `--gap` controls the endpoint interval and `--pair-stride` (alias `--stride`) controls the step between starting frames. Only complete pairs are labeled; endpoints are never padded or repeated. An N-frame interval produces `floor((N-1)/4)` rows with these defaults (at least five frames are required). More generally, the count is `max(0, floor((N-1-gap)/pair_stride)+1)`.

NPZ output contains:

- `latent_action`: `[num_pairs, K*D]`, float16; selected with `--representation`.
- `pair_indices`: `[num_pairs, 2]`, int32, episode-local source/target RGB frame indices, not latent row indices.
- `meta`: JSON with checkpoint SHA-256, code shape, normalization, geometry, FPS, true decoded `num_frames`, `unpaired_tail_frames`, and the protocol's `gap`/`pair_stride`.

Sampling starts at frame zero of each declared episode interval; add `protocol.start_frame` to `pair_indices` to recover packed-video indices. For Wan's common causal 4x temporal layout, `4L+1` RGB frames correspond to an initial conditioning video latent plus L subsequent latents. This script emits L motion intervals; it does not invent a motion label for the initial conditioning frame. For example, 97 RGB frames yield 24 actions and typically 25 video latents. The generator must align these explicitly and handle the unused tail consistently.

To request dense labels instead, set `--pair-stride 1` and the desired gap. Temporal stride is part of protocol version 2: old dense labels cannot silently satisfy a sparse-label resume. Use a separate output directory. Stride four reduces model pair evaluations and output volume by approximately four, but compressed-video decoding and source IO do not necessarily become four times faster.

For multiple episodes use `--metadata episodes.json`, a list of `{dataset, episode_id, caption, views}` records; each view requires `video_path` and may contain `start_frame`/`end_frame`. Relative video paths are resolved against the metadata file. `--views head wrist_left` filters views. `--shard 0 --num-shards 8` partitions episodes; run each shard once with the same inputs.

Complete labels are published atomically and checked before resume. Checkpoint, source, and implementation fingerprints bind each labeling protocol. A mismatch fails unless you explicitly choose `--overwrite` or a new output directory. `--rebuild-index` validates existing labels without inference. A failed view returns a nonzero exit status and is recorded in `failed_shardNNNN.json`. Use a local `--scratch` directory with space for one video's uncompressed token arrays and compressed NPZ; tokens are buffered on disk and validated in chunks rather than accumulated in RAM.

**Normalization matters:** release inference, labeling, and extraction default to ImageNet normalization. The historical extractor could use unnormalized inputs. To reproduce that specific protocol, use `--normalization none` (or `normalize=False` in Python), and report it explicitly. Scores from different normalization, geometry, pooling, or action-statistics protocols must not be mixed. Old benchmark scores are not claimed for this release.

PyTorch checkpoints are loaded with `weights_only=True` by default. The optional inference flag `--trust-checkpoint` enables Python-object deserialization and must only be used with a checkpoint you trust.

## Latent action benchmark

`evaluate.py` supports CALVIN, VLABench, RoboCOIN, and AgiBot. It first freezes RynnLAM and extracts features; a separate supplied LARYBench checkout trains the action-regression probe. Action labels are used by the benchmark probe, not by RynnLAM itself.

```bash
python evaluate.py extract --dataset calvin --split train \
  --csv /path/to/calvin_train.csv --tar-dir /path/to/calvin_tars \
  --tar-glob 'train_stride5_part*.tar.gz' \
  --checkpoint /path/to/checkpoint.pt --output-root ./outputs/bench \
  --representation ktoken --pool flat --resume
```

Repeat for the validation split with its CSV and archives. Extraction prints a `report.json` location. Compressed tar archives are each decompressed once to local scratch before indexed reads; allow space for the largest uncompressed archive and set `--local-temp-dir` when needed. For multi-shard extraction, merge all shard reports with `evaluate.py merge --reports ... --output-dir ...`; full row coverage is required before regression.

```bash
python evaluate.py regress --lary-root /path/to/LARYBench \
  --train-report /path/to/train/report.json \
  --val-report /path/to/val/report.json --output-root ./outputs/probes
```

Benchmark crops default to fixed 238×322, independent of any training bucket choice. Default strides are 5/5/10/45 for CALVIN/VLABench/RoboCOIN/AgiBot. Extraction preserves `[K,D]` for a flat probe; the regression loader flattens it. Mean pooling is an explicit alternative. Checkpoint, input metadata, protocol, and regression settings are recorded, and separate output directories prevent concurrent jobs from appending to one shared result file. Incomplete extraction or regression is not reported as success.

Install the supplied LARYBench checkout's dependencies separately. This wrapper targets its `regression/main.py` interface and absolute-action protocol; it does not bundle or claim to reproduce every historical internal evaluation variant. Use identical train/validation splits and probe settings for model comparisons. Lower probe MSE does not by itself establish higher downstream VLA success.

## Tests

```bash
python -m pytest -q
```

Tests exercise synthetic video labeling and resume, manifest recovery, bucket/role sampling, checkpoint strictness, and benchmark coverage. Optional original-source numerical parity checks require `RYNNLAM_BASELINE_ROOT`; ordinary tests do not require the original repository. GPU and real-data throughput depend on your environment.

## License

RynnLAM source is released under Apache-2.0. See `LICENSE` and `NOTICE` for retained third-party attribution. Model weights, datasets, and externally installed benchmark software have separate licenses and are not covered by this source release.
