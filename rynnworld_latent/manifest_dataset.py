"""Manifest-driven RynnWorld-Latent dataset (no materialized episodes).

Reads chunk records from the manifest (``chunks_*.jsonl``, e.g. the bundled
``data/manifest``) and loads, per sample, ``nf`` frames from a video at a frame
offset plus ``na`` latent actions
from the paired ``latent.npz``. This is the full-scale path: at ~616k chunks,
materializing one mp4 per chunk would cost ~1.8 TB and ~1.9M files on OSS FUSE.

Actions are RynnLAM ``ktoken_zcam`` latents: one npz key ``latent_action [T,608]``
laid out as ``[k_tokens 512 | z 64 | camera 32]``, labeled with gap=4 /
pair_stride=4, so they are already at action rate and a chunk takes ``na``
consecutive latents from ``lat_start = start//4``. Absolute video frames start at
``start_frame`` from the npz ``meta`` (packed sources label a sub-interval of a
shared video), and non-file sources (``rovidx_tar://`` / zarr / hdf5) decode
through RynnLAM's FrameReader (see ``rynnworld_latent.rynnlam_bridge``).

Two video sources are supported:

* ``staged_root`` set  -> read ``<staged_root>/<rec.vid>`` (your own pre-staged
  480p copies). Useful on a cluster that cannot see the original source mounts.
* ``staged_root`` None -> read ``rec.src`` (the original source video). The
  default, and what the single-node quickstart uses.

Decode failures are expected at this scale (a few unreadable/short videos), so a
failing sample deterministically falls back to other records instead of crashing
the training job.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rynnworld_latent import rynnlam_bridge
from rynnworld_latent.dataset import (
    ACTION_DIM,
    DOMAIN_ID,
    load_action_stats,
    load_video_chunk_av,
    load_video_chunk_framereader,
    normalize_action,
)


_TRANSIENT_ERRNOS = frozenset(
    {errno.ENOTCONN, errno.EIO, errno.EINTR, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN}
)


def _transient_io(e: Exception) -> bool:
    """True for errors a wait can survive (FUSE mount dropout), false for bad media."""
    if isinstance(e, ConnectionError):
        return True
    return getattr(e, "errno", None) in _TRANSIENT_ERRNOS


class RynnWorldManifestDataset(Dataset):
    """Map-style dataset over manifest chunk records."""

    def __init__(
        self,
        manifest_dir: str,
        staged_root: str | None = None,
        fps: float | None = None,
        mode: str = "forward_dynamics",
        action_normalization: str | None = "quantile",
        viewpoint: str = "ego_view",
        stats_path: str | None = None,
        limit: int = 0,
        max_retries: int = 8,
        src_remap: str | None = None,
    ) -> None:
        super().__init__()
        self._manifest_dir = Path(manifest_dir)
        self._staged_root = Path(staged_root) if staged_root else None
        self._fps_override = fps
        self._mode = mode
        self._viewpoint = viewpoint
        self._action_normalization = action_normalization
        self._max_retries = int(max_retries)

        # Source-path remapping: the manifest records absolute paths as seen on the
        # box that built it. On the training machine those buckets/NAS may mount
        # elsewhere, so allow prefix rewrites via "from=>to,from2=>to2".
        raw_remap = src_remap if src_remap is not None else os.environ.get("RYNNWORLD_SRC_REMAP", "")
        self._src_remap: list[tuple[str, str]] = []
        for rule in (raw_remap or "").split(","):
            rule = rule.strip()
            if not rule or "=>" not in rule:
                continue
            src_pref, _, dst_pref = rule.partition("=>")
            src_pref, dst_pref = src_pref.strip(), dst_pref.strip()
            if src_pref and dst_pref:
                self._src_remap.append((src_pref, dst_pref))
        if self._src_remap:
            print(f"[RynnWorldManifestDataset] src remap rules: {self._src_remap}")

        shards = sorted(self._manifest_dir.glob("chunks_*.jsonl"))
        if not shards:
            raise RuntimeError(f"no chunks_*.jsonl under {manifest_dir}")

        # Bundled release samples ship paths relative to the repo data/ dir so the
        # quickstart trains on any machine; full-scale manifests stay absolute.
        bundle_root = self._manifest_dir.parent

        # Compact in-memory records. Strings are interned because latent paths,
        # captions and video paths repeat across the chunks of one view/episode;
        # at 616k records this materially reduces per-worker RSS.
        self._recs: list[tuple] = []
        for shard in shards:
            with open(shard) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("as") != "v2":
                        raise ValueError(
                            f"{shard}: record has action schema {r.get('as')!r}, expected "
                            "'v2'; this dataset only reads RynnLAM ktoken_zcam manifests"
                        )
                    self._recs.append(
                        (
                            sys.intern(r["vid"]),
                            sys.intern(r["src"] if os.path.isabs(r["src"]) else str(bundle_root / r["src"])),
                            sys.intern(r["lat"] if os.path.isabs(r["lat"]) else str(bundle_root / r["lat"])),
                            int(r["start"]),
                            int(r["nf"]),
                            int(r["na"]),
                            sys.intern(r["cap"]),
                            float(r["fps"]),
                            int(r["lat_start"]),
                            int(r["lat_stride"]),
                            sys.intern(r.get("view", "")),
                            sys.intern(r.get("ds", "")),
                        )
                    )
                    if limit and len(self._recs) >= limit:
                        break
            if limit and len(self._recs) >= limit:
                break

        # Normalization stats: optional, but when present their dimension must
        # match the action width — normalizing 608-dim actions with stats from a
        # different labeler would silently corrupt every sample.
        self._norm_stats: dict[str, np.ndarray] | None = None
        if self._action_normalization is not None:
            sp = Path(stats_path) if stats_path else (self._manifest_dir / "action_stats.json")
            if sp.exists():
                self._norm_stats = load_action_stats(sp)
                probe = next(iter(self._norm_stats.values()))
                if probe.shape[0] != ACTION_DIM:
                    raise RuntimeError(
                        f"{sp} is {probe.shape[0]}-dim but RynnLAM ktoken_zcam actions are "
                        f"{ACTION_DIM}-dim; the manifest's action_stats.json must be "
                        f"{ACTION_DIM}-dim (see the bundled data/manifest for the format)"
                    )
            else:
                # Missing stats is never intentional here: the unnormalized path is
                # action_normalization=None (excluded by the outer if). Training on raw
                # latents (~±20 instead of [-1,1]) silently corrupts the run while the
                # loss still looks finite, and this mount has lost written files before
                # — fail fast instead of warning into a 6-worker log flood.
                raise RuntimeError(
                    f"no action stats at {sp} but action_normalization="
                    f"{self._action_normalization!r}: actions would train UNNORMALIZED "
                    "(raw latents ~±20 instead of [-1,1]). The manifest must ship a "
                    f"{ACTION_DIM}-dim action_stats.json (as the bundled data/manifest does), "
                    "or pass action_normalization=None if unnormalized is intentional."
                )
        self._warned_stats_dim = False

    def __len__(self) -> int:
        return len(self._recs)

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _remap(self, path: str) -> str:
        """Apply the first matching prefix rewrite rule (no-op when none match)."""
        for src_pref, dst_pref in self._src_remap:
            if path.startswith(src_pref):
                return dst_pref + path[len(src_pref) :]
        return path

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._norm_stats is None or self._action_normalization is None:
            return action.astype(np.float32)
        probe = next(iter(self._norm_stats.values()))
        if probe.shape[0] != action.shape[-1]:
            if not self._warned_stats_dim:
                print(
                    f"[RynnWorldManifestDataset] WARNING: action_stats dim {probe.shape[0]} != "
                    f"action dim {action.shape[-1]}; skipping normalization."
                )
                self._warned_stats_dim = True
            return action.astype(np.float32)
        return normalize_action(action, self._norm_stats, self._action_normalization)

    def _load_one(self, i: int) -> dict[str, Any]:
        vid, src, lat, start, nf, na, cap, fps, lat_start, lat_stride, view, _ds = self._recs[i]

        # Actions first: the npz meta carries start_frame, which the video decode
        # needs, and a bad latent file fails fast before any video download.
        with np.load(self._remap(lat)) as d:  # numeric arrays + meta; no pickle
            latent_all = d["latent_action"]
            idx = [lat_start + j * lat_stride for j in range(na)]
            if idx[-1] >= latent_all.shape[0]:
                raise IndexError(f"latent index {idx[-1]} >= {latent_all.shape[0]} in {lat}")
            if latent_all.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"expected {ACTION_DIM}-dim latent_action in {lat}, "
                    f"got {latent_all.shape[-1]}"
                )
            start_frame = rynnlam_bridge.read_npz_start_frame(d, lat)
            action = latent_all[idx].astype(np.float32)  # [na, 608]
        action = self._normalize_action(action)

        # Resolve the video source: an optional pre-staged local copy (staged_root)
        # takes precedence over the remapped original source path.
        staged_path: str | None = None
        if self._staged_root is not None:
            s = self._staged_root / vid
            if s.exists():
                staged_path = str(s)
        vpath = staged_path if staged_path is not None else self._remap(src)
        # Absolute decode offset: packed sources (e.g. Droid) label a sub-interval
        # of a shared video, so the chunk starts at start_frame + start.
        abs_start = start_frame + start
        if rynnlam_bridge.needs_framereader(vpath):
            video = load_video_chunk_framereader(vpath, view or "head", abs_start, nf)
        else:
            # PyAV (libdav1d) for plain files: the corpus mixes AV1 sources the
            # local cv2 build cannot decode; same decoder RynnLAM labeled with.
            video = load_video_chunk_av(Path(vpath), abs_start, nf)

        return {
            "ai_caption": cap,
            "video": video.permute(1, 0, 2, 3).contiguous(),  # [C,T,H,W] uint8
            "action": torch.from_numpy(action),  # [na, 608]
            "conditioning_fps": torch.tensor(
                int(self._fps_override if self._fps_override is not None else fps), dtype=torch.long
            ),
            "mode": self._mode,
            "domain_id": torch.tensor(DOMAIN_ID, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self._recs)
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            i = (idx + attempt * 7919) % n  # coprime stride -> spreads retries
            try:
                return self._load_one(i)
            except Exception as e:  # noqa: BLE001 - bad media must not kill training
                last_err = e
                if attempt == 0:
                    print(f"[RynnWorldManifestDataset] sample {i} failed ({type(e).__name__}: {e}); retrying")
                if _transient_io(e):
                    wait = min(90, 5 * 2**attempt)
                    print(f"[RynnWorldManifestDataset] transient I/O error on sample {i}; "
                          f"sleeping {wait}s before retry {attempt + 1}/{self._max_retries}")
                    time.sleep(wait)
        raise RuntimeError(f"all {self._max_retries} retries failed near idx {idx}: {last_err}")