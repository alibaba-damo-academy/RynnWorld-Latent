"""Video decoding and latent-action normalization for RynnWorld-Latent.

The data contract is RynnLAM's ``ktoken_zcam`` latent action: ONE npz key
``latent_action [T, 608]`` laid out as ``[k_tokens 512 | z 64 | camera 32]``,
labeled with gap=4 / pair_stride=4, so latent ``n`` covers RGB frames
``[4n, 4n+4]`` and is already at action rate — one latent per 4 RGB frames, which
is exactly one Wan-VAE latent frame. A training chunk is therefore 81 pixel
frames (1 conditioning + 80 future) paired with 20 consecutive latents.

Chunk records (the ``chunks_*.jsonl`` manifest, e.g. the bundled ``data/manifest``)
are consumed by ``rynnworld_latent/manifest_dataset.py``. This module holds the
pieces they share: resolution-tier bucketing, the three video readers, and action
normalization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

# Cosmos3-Edge action domain 3 = hand_pose: the most-specialized pretrained
# action domain (highest per-domain weight norms) and semantically closest to
# first-person human-hand video, which is what the corpus is. Every sample is
# tagged with it; the action pathway itself is zero-initialized (scripts/train.py).
DOMAIN_ID = 3

# RynnLAM ktoken_zcam latent action: [k_tokens 512 | z 64 | camera 32].
# See rynnworld_latent/rynnlam_bridge.py.
ACTION_DIM = 608

# Resolution-tier buckets, mirrored from cosmos ``VIDEO_RES_SIZE_INFO``
# (cosmos_framework/data/generator/utils.py) so the decode-time downscale routes
# each video to the SAME (w, h) bucket the model's data transform will pick via
# ``find_closest_target_size``. Values are (target_w, target_h) per aspect ratio.
_RES_BUCKETS = {
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
}
# Which tier to bucket to at decode time; MUST match the model config ``resolution``.
# "", "0" or "none" disables the decode-time downscale (debug only).
_RESOLUTION_TIER = os.environ.get("RYNNWORLD_RESOLUTION", "480").strip()
# An unknown tier would make _closest_bucket return None, silently turning the
# decode-time downscale off while the model's transform still uses its own
# resolution — the two geometries desync with no error anywhere. Fail fast.
if _RESOLUTION_TIER.lower() not in ("", "0", "none") and _RESOLUTION_TIER not in _RES_BUCKETS:
    raise RuntimeError(
        f"RYNNWORLD_RESOLUTION={_RESOLUTION_TIER!r} is not a known resolution tier "
        f"{sorted(_RES_BUCKETS)} (or '' to disable the decode-time downscale)"
    )


def _closest_bucket(h: int, w: int, tier: str) -> tuple[int, int] | None:
    """Return (target_w, target_h) of the tier bucket whose aspect ratio is closest to h/w."""
    table = _RES_BUCKETS.get(tier)
    if not table or h <= 0 or w <= 0:
        return None
    in_ratio = h / w
    best, best_d = None, float("inf")
    for cw, ch in table.values():
        d = abs((ch / cw) - in_ratio)
        if d < best_d:
            best, best_d = (cw, ch), d
    return best


# Keys in action_stats.json that are arrays (used for normalization)
_NORM_KEYS = {"q01", "q99", "mean", "std"}


def load_action_stats(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``action_stats.json`` into float32 arrays, keeping only ``_NORM_KEYS``.

    Scalar bookkeeping keys are dropped so callers can probe the action dimension
    with ``next(iter(stats.values())).shape[0]``.
    """
    raw = json.loads(Path(path).read_text())
    return {
        key: np.array(value, dtype=np.float32)
        for key, value in raw.items()
        if key in _NORM_KEYS and isinstance(value, list)
    }


def normalize_action(
    action: np.ndarray,
    stats: dict[str, np.ndarray] | None,
    mode: str | None = "quantile",
) -> np.ndarray:
    """Normalize a ``[..., action_dim]`` action array with precomputed statistics.

    ``"quantile"`` maps the q01..q99 range onto [-1, 1] and clips; ``"zscore"``
    standardizes. A per-dimension denominator below 1e-8 falls back to 1.0, so a
    constant channel stays constant instead of dividing by zero. Any other mode —
    including None, or missing stats — returns the action unchanged as float32.

    Callers are responsible for checking that the stats dimension matches the
    action's channel count; broadcasting mismatched stats would fail loudly.
    """
    if stats is None or mode is None:
        return np.asarray(action, dtype=np.float32)
    if mode == "quantile":
        q01, q99 = stats["q01"], stats["q99"]
        denom = np.where(np.abs(q99 - q01) < 1e-8, 1.0, q99 - q01)
        return np.clip(2.0 * (action - q01) / denom - 1.0, -1.0, 1.0).astype(np.float32)
    if mode == "zscore":
        std = np.where(stats["std"] < 1e-8, 1.0, stats["std"])
        return ((action - stats["mean"]) / std).astype(np.float32)
    return np.asarray(action, dtype=np.float32)


def _tier_target_wh(fh: int, fw: int) -> tuple[int, int]:
    """Decode-time downscale target for a (h, w) frame under the active tier.

    Returns (target_w, target_h), or (0, 0) when no resize should happen (tier
    disabled, unknown aspect, or the source is already small enough).
    """
    if _RESOLUTION_TIER.lower() in ("", "0", "none"):
        return (0, 0)
    bucket = _closest_bucket(fh, fw, _RESOLUTION_TIER)
    if bucket is None:
        return (0, 0)
    btw, bth = bucket
    # Downscale only, aspect-preserving, so neither side drops below the
    # bucket target (cosmos's aspect-resize + reflection-pad then fits it
    # exactly without upscaling). Never enlarge already-small sources.
    scale = max(bth / fh, btw / fw)
    if scale >= 1.0:
        return (0, 0)
    return (max(2, round(fw * scale / 2) * 2), max(2, round(fh * scale / 2) * 2))


def _tier_resize(frame, wh: tuple[int, int]):
    """Apply a ``_tier_target_wh`` result to one frame (no-op when wh == (0, 0))."""
    if wh[0]:
        import cv2

        return cv2.resize(frame, wh, interpolation=cv2.INTER_AREA)
    return frame


def load_video_chunk_av(video_path: Path, start: int, length: int) -> torch.Tensor:
    """Seek-based PyAV decode of frames ``[start, start+length)`` -> [N,C,H,W] uint8.

    Plain-file sources go through PyAV instead of cv2: parts of the corpus
    are AV1-encoded (e.g. AgiBot), which the local OpenCV/ffmpeg build cannot
    decode at all (every read fails, and even opening spams the log), while
    PyAV bundles libdav1d — the same decoder RynnLAM labeled with, so pixels
    match the labeler's view of the video. The seek lands on the keyframe
    before the target and frames are positioned by pts (corpus is CFR), so a
    mid-video chunk costs ~GOP + length decodes, not the whole prefix.
    """
    import av

    frames = []
    resize_wh: tuple[int, int] | None = None
    native_hw: tuple[int, int] | None = None
    with av.open(str(video_path)) as ct:
        stream = ct.streams.video[0]
        # "NONE", not "AUTO": ffmpeg's frame-threaded decoder builds a ~5-13 thread
        # pool per container, and breaking out of the decode loop once we have our
        # frames means those pools are never reclaimed. Measured on a 26-core box:
        # one dataloader worker accumulated 190-396 live `av:hevc:df*` threads and
        # retained ~26 MiB of host RAM per sample, which OOM-killed a run at
        # iteration 3489. Single-threaded decode retains nothing (20 decodes:
        # threads 1 -> 1, RSS flat) and costs 0.18-0.25 s instead of 0.06-0.12 s per
        # 81-frame chunk -- irrelevant against a ~25 s iteration budget that 8
        # workers share.
        stream.thread_type = "NONE"
        if not stream.average_rate:
            raise RuntimeError(f"cannot determine fps for seek decode: {video_path}")
        fps = float(stream.average_rate)
        if start > 0:
            # 1e6 is AV_TIME_BASE (the timebase of ffmpeg's seek offset), spelled out
            # rather than `av.time_base.denominator`: some PyAV versions expose
            # time_base as the plain int 1000000, whose .denominator is 1, making the
            # seek offset 1e6 times too small -- the seek silently lands on frame 0 and
            # the pts-discard loop below decodes the whole prefix (cost grows with
            # `start`, no error raised). Hardcoding AV_TIME_BASE is correct on every
            # version.
            ct.seek(int(start / fps * 1_000_000), backward=True, any_frame=False)
        for frame in ct.decode(video=0):
            if frame.time is None:
                raise RuntimeError(f"frame without pts while seeking in {video_path}")
            if int(round(frame.time * fps)) < start:
                continue  # keyframe-before-target lead-in
            rgb = frame.to_ndarray(format="rgb24")
            hw = rgb.shape[:2]
            if native_hw is None:
                native_hw = hw
                resize_wh = _tier_target_wh(*hw)
            elif hw != native_hw:
                raise RuntimeError(
                    f"frame resolution changed mid-clip in {video_path}: {native_hw} -> {hw}"
                )
            frames.append(torch.from_numpy(_tier_resize(rgb, resize_wh)).permute(2, 0, 1))
            if len(frames) >= length:
                break
    if len(frames) < length:
        raise RuntimeError(
            f"Video {video_path} decoded {len(frames)} < {length} frames from start={start}"
        )
    return torch.stack(frames)  # [N, C, H, W] uint8


def load_video_chunk_framereader(
    src: str, view: str, start: int, length: int
) -> torch.Tensor:
    """Load ``[start, start+length)`` frames via RynnLAM's FrameReader.

    For sources cv2 cannot open (``rovidx_tar://`` URLs, zarr directories,
    hdf5 files); reuses the exact decoder the labels were produced with.
    FrameReader yields native-resolution RGB and raises on an incomplete
    interval, so only the tier downscale is applied here — identical geometry
    to ``load_video_chunk_av``. Returns [N, C, H, W] uint8.
    """
    from rynnworld_latent import rynnlam_bridge

    video_mod = rynnlam_bridge.ensure_importable()
    reader = video_mod.FrameReader(src, view)
    frames = []
    resize_wh: tuple[int, int] | None = None
    native_hw: tuple[int, int] | None = None
    for frame_rgb in reader.frames(start, start + length):
        hw = frame_rgb.shape[:2]
        if native_hw is None:
            native_hw = hw
            resize_wh = _tier_target_wh(*hw)
        elif hw != native_hw:
            # FrameReader.frames does not check mid-clip resolution changes and
            # torch.stack would fail obscurely; name the real problem.
            raise RuntimeError(
                f"frame resolution changed mid-clip in {src}: {native_hw} -> {hw}"
            )
        frame = _tier_resize(frame_rgb, resize_wh)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # [C, H, W]
    return torch.stack(frames)  # [N, C, H, W] uint8


