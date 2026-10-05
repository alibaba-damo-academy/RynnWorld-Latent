"""Lazy bridge to the bundled RynnLAM checkout.

RynnLAM labels videos with a single 608-dim ``latent_action`` key
(``[k_tokens 512 | z 64 | camera 32]``, gap=4 / pair_stride=4), and parts of the
RynnVLA-Base corpus are not plain video files: ``rovidx_tar://`` URLs (byte-offset
reads into tar shards), zarr directories (EgoVerse) and hdf5 files (RoboMIND).
``rynnlam.video.FrameReader`` already implements every one of those backends and
is the exact decoder the labels were produced with, so video loading reuses it
instead of reimplementing.

Everything here is imported lazily: a run over plain video files never touches
RynnLAM or its av/zarr/h5py dependencies.

Env:
    RYNNLAM_ROOT          RynnLAM checkout override. Defaults to the bundled
                          top-level ``rynnlam/`` directory in this repo.
    ROVIDX_TAR_ROOT       tar shard root for rovidx_tar:// sources.
    ROVIDX_INDEX_ROOT     jsonl byte-offset index root (both required only when
                          a rovidx_tar:// source is read; rynnlam.video raises a
                          clear error if they are missing).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

_ROVIDX_PREFIX = "rovidx_tar://"
# Resolution order: RYNNLAM_ROOT env override > the bundled top-level rynnlam/
# checkout, which keeps the repo self-contained.
_BUNDLED_RYNNLAM_ROOT = str(Path(__file__).resolve().parent.parent / "rynnlam")

_video_module = None


def _force_single_threaded_av() -> None:
    """Make every ``av.open`` in this process decode single-threaded.

    ``rynnlam.video.FrameReader`` opens its own containers (``rovidx_tar://``
    members and plain files) and never sets ``thread_type``, so ffmpeg
    auto-enables frame threading: each container builds a ~5-13 thread decoder
    pool that is never reclaimed once the frame loop is exited early. Measured on
    the rovidx_tar path: threads 26 -> 250 over 48 samples, ~5.7 MiB of host RAM
    retained per sample. RynnLAM is an external checkout we do not modify, so
    patch ``av.open`` here instead -- ``rynnlam.video`` imports ``av`` lazily
    inside its functions, so it picks up the patched attribute.
    """
    try:
        import av
    except Exception:  # noqa: BLE001 - av missing is reported by the caller's backend
        return
    if getattr(av, "_rynnworld_single_threaded", False):
        return
    real_open = av.open

    def open_single_threaded(*args, **kwargs):
        container = real_open(*args, **kwargs)
        try:
            for stream in container.streams.video:
                stream.thread_type = "NONE"
        except Exception:  # noqa: BLE001 - never let a perf knob break a read
            pass
        return container

    av.open = open_single_threaded
    av._rynnworld_single_threaded = True


def ensure_importable():
    """Import and return ``rynnlam.video``, adding RYNNLAM_ROOT to sys.path once."""
    global _video_module
    if _video_module is not None:
        return _video_module
    root = os.environ.get("RYNNLAM_ROOT") or _BUNDLED_RYNNLAM_ROOT
    if not (Path(root) / "rynnlam" / "video.py").is_file():
        raise RuntimeError(
            f"RYNNLAM_ROOT={root!r} does not contain rynnlam/video.py; point it at a "
            "RynnLAM checkout to read the special-format sources"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from rynnlam import video as _video

    _force_single_threaded_av()
    _video_module = _video
    return _video_module


# Suffixes that decide the answer with no I/O at all. A zarr source presents as a
# directory with no video suffix, so anything ending in one of these is a plain file.
_PLAIN_VIDEO_SUFFIXES = frozenset({
    ".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".ts", ".flv",
})
_HDF5_SUFFIXES = frozenset({".hdf5", ".h5"})


def needs_framereader(src: str) -> bool:
    """True when ``src`` needs RynnLAM's FrameReader rather than a plain video decode.

    Mirrors ``rynnlam.video.FrameReader``'s backend dispatch: rovidx_tar URL,
    directory (zarr) or hdf5 suffix need the FrameReader; everything else stays
    on the seekable PyAV path (``dataset.load_video_chunk_av``).

    Decides from the path string alone whenever the string is sufficient, and only
    stats as a last resort. Both reasons are load-bearing:

    * Object-store FUSE mounts abort connections (``Errno 103 Software caused
      connection abort``, observed on a plain ``.mp4``), and ``Path.is_dir()``
      raises there. ``manifest_dataset.py`` has a retry loop, but one hiccup
      would still burn a sample.
    * ``manifest_dataset.py`` calls this for EVERY training sample, so an
      unconditional ``is_dir()`` is one ~45 ms FUSE stat per sample to answer a
      question the suffix already answered.
    """
    text = str(src)
    if text.startswith(_ROVIDX_PREFIX):
        return True
    suffix = Path(text).suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        return True
    if suffix in _PLAIN_VIDEO_SUFFIXES:
        return False

    # Suffix-less: this is how a zarr directory presents, so the stat is the only
    # way to know -- and guessing wrong would hand a directory to a video decoder.
    path = Path(text)
    last: OSError | None = None
    for attempt in range(3):
        try:
            return path.is_dir()
        except OSError as e:
            last = e
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(
        f"cannot stat {text!r} to decide the video backend ({type(last).__name__}: {last}). "
        f"Refusing to guess: treating a zarr directory as a plain file (or the reverse) "
        f"fails silently instead of loudly."
    ) from last


# latent path -> start_frame. Chunks of one view share the same npz, and the
# meta parse is per-sample, so a small LRU keeps it off the hot path.
_START_FRAME_CACHE: OrderedDict[str, int] = OrderedDict()
_START_FRAME_CACHE_MAX = 4096


def read_npz_start_frame(data, path_key: str) -> int:
    """Absolute video frame that latent index 0 refers to (npz ``meta`` key).

    The RynnLAM index does not carry start_frame — it only lives in the npz
    ``meta.protocol``. Packed sources (e.g. Droid episodes sharing one long mp4)
    label a sub-interval, so the absolute frame of latent n is
    ``start_frame + n * pair_stride`` and video chunks must decode from
    ``start_frame + start``. Missing meta is fatal, not 0: silently assuming 0
    would misalign every packed-dataset sample.
    """
    cached = _START_FRAME_CACHE.get(path_key)
    if cached is not None:
        _START_FRAME_CACHE.move_to_end(path_key)
        return cached
    if "meta" not in data:
        raise ValueError(
            f"{path_key}: latent.npz has no 'meta' key; cannot resolve start_frame"
        )
    meta = json.loads(str(data["meta"].item()))
    start_frame = meta.get("protocol", {}).get("start_frame", None)
    if start_frame is None:
        raise ValueError(f"{path_key}: meta.protocol has no start_frame")
    start_frame = int(start_frame)
    _START_FRAME_CACHE[path_key] = start_frame
    if len(_START_FRAME_CACHE) > _START_FRAME_CACHE_MAX:
        _START_FRAME_CACHE.popitem(last=False)
    return start_frame
