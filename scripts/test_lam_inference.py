#!/usr/bin/env python3
"""RynnLAM inference check: re-encode the bundled videos and compare to the shipped latents.

This proves two things at once:

* RynnLAM inference actually runs in this environment — the checkpoint loads, the
  encoder forward pass works, and the ``ktoken_zcam`` representation has the
  expected 608-dim ``[k_tokens 512 | z 64 | camera 32]`` layout.
* The latents bundled under ``data/latents/`` are **reproducible** from
  ``data/videos/`` using the exact checkpoint recorded in each npz's
  ``meta.protocol.checkpoint_sha256``. A bundled sample whose labels cannot be
  regenerated is a sample nobody can trust or extend.

The geometry is taken from each npz's own ``meta`` (``bucket`` / ``target_hw``)
and cross-checked against ``rynnlam.video.pick_bucket``, so a silent bucket
mismatch shows up as a failure rather than as slightly-wrong numbers.

Usage::

    python scripts/test_lam_inference.py --lam-ckpt /path/to/lam_b512.pt
    RYNNLAM_CKPT=/path/to/lam_b512.pt python scripts/test_lam_inference.py
    python scripts/test_lam_inference.py --lam-ckpt ... --max-pairs 8   # quick

Exit status is non-zero if any episode falls outside ``--rel-tol`` / ``--cos-tol``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

K_SLICE = slice(0, 512)
Z_SLICE = slice(512, 576)
CAM_SLICE = slice(576, 608)


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def bundle_video_for(data_dir: Path) -> dict[str, Path]:
    """Map each bundled latent.npz to its source video, via the manifest.

    The manifest is the authority: its ``lat`` and ``src`` fields are the pairing
    the training path actually uses, and both are relative to ``data/``.
    """
    out: dict[str, Path] = {}
    shards = sorted((data_dir / "manifest").glob("chunks_*.jsonl"))
    if not shards:
        raise SystemExit(f"no manifest shards under {data_dir / 'manifest'}")
    for shard in shards:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lat = r["lat"] if os.path.isabs(r["lat"]) else str(data_dir / r["lat"])
            src = r["src"] if os.path.isabs(r["src"]) else str(data_dir / r["src"])
            out.setdefault(lat, Path(src))
    return out


def read_frames(reader, needed: set[int]) -> dict[int, np.ndarray]:
    """Decode only the frames in ``needed`` (FrameReader yields sequentially)."""
    got: dict[int, np.ndarray] = {}
    top = max(needed)
    for idx, frame in enumerate(reader.frames(0, top + 1)):
        if idx in needed:
            got[idx] = frame
        if idx >= top:
            break
    missing = needed - set(got)
    if missing:
        raise RuntimeError(f"video ended early; missing frames {sorted(missing)[:5]}")
    return got


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lam-ckpt", default=os.environ.get("RYNNLAM_CKPT", ""),
                    help="RynnLAM checkpoint (default: $RYNNLAM_CKPT)")
    ap.add_argument("--data-dir", default=str(_REPO / "data"),
                    help="bundled data dir holding videos/ latents/ manifest/ (default: <repo>/data)")
    ap.add_argument("--episodes", default="", help="comma-separated latent dir names; default: all")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap pairs per episode (0 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rel-tol", type=float, default=0.05,
                    help="max mean|diff| / mean|stored| per episode")
    ap.add_argument("--cos-tol", type=float, default=0.999, help="min mean per-row cosine")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--skip-sha256", action="store_true",
                    help="skip hashing the checkpoint (it can be GBs on a slow mount)")
    args = ap.parse_args()

    if not args.lam_ckpt:
        ap.error("--lam-ckpt is required (or set RYNNLAM_CKPT)")
    ckpt = Path(args.lam_ckpt).expanduser()
    if not ckpt.is_file():
        raise SystemExit(f"LAM checkpoint not found: {ckpt}")
    data_dir = Path(args.data_dir).expanduser().resolve()

    from rynnworld_latent import rynnlam_bridge

    video_mod = rynnlam_bridge.ensure_importable()
    from rynnlam.inference import RynnLAMEncoder

    print(f"[lam] checkpoint : {ckpt}")
    print(f"[lam] data dir   : {data_dir}")
    print(f"[lam] RynnLAM    : {Path(video_mod.__file__).parent}")

    sha = None
    if not args.skip_sha256:
        print("[lam] hashing checkpoint (this can take a while on a network mount) ...", flush=True)
        sha = file_sha256(ckpt)
        print(f"[lam] sha256     : {sha}")

    videos = bundle_video_for(data_dir)
    lat_dirs = sorted((data_dir / "latents").glob("*/latent.npz"))
    if args.episodes:
        want = set(args.episodes.split(","))
        lat_dirs = [p for p in lat_dirs if p.parent.name in want]
    if not lat_dirs:
        raise SystemExit(f"no bundled latent.npz found under {data_dir / 'latents'}")

    encoder = RynnLAMEncoder(ckpt, device=args.device, normalize=True,
                             precision=args.precision, checkpoint_sha256=sha or "")
    print(f"[lam] encoder ready on {args.device} ({args.precision})")
    print("[lam] note: RynnLAM logs 'No encoder checkpoint supplied; using random")
    print("[lam]       initialization' while *constructing* the DINOv2 backbone;")
    print("[lam]       load_checkpoint then overwrites the full state dict, so the")
    print("[lam]       warning is benign. The comparison below is the real check.\n")

    failures: list[str] = []
    for lat_path in lat_dirs:
        name = lat_path.parent.name
        with np.load(lat_path) as d:
            stored = d["latent_action"].astype(np.float32)
            pairs = d["pair_indices"].astype(np.int64)
            meta = json.loads(str(d["meta"]))
        proto = meta.get("protocol", {})
        vpath = videos.get(str(lat_path))
        if vpath is None or not vpath.is_file():
            failures.append(f"{name}: no bundled video for this latent (manifest pairing missing)")
            print(f"=== {name}: SKIP (no video)\n")
            continue

        # --- provenance + geometry self-checks -----------------------------
        probs = []
        if sha and proto.get("checkpoint_sha256") and sha != proto["checkpoint_sha256"]:
            probs.append(f"checkpoint sha256 {sha[:12]}... != meta {proto['checkpoint_sha256'][:12]}...")
        if proto.get("gap") != 4 or proto.get("pair_stride") != 4:
            probs.append(f"meta gap/pair_stride = {proto.get('gap')}/{proto.get('pair_stride')}, expected 4/4")
        if proto.get("representation") != "ktoken_zcam":
            probs.append(f"meta representation = {proto.get('representation')!r}")
        if stored.shape[1] != 608:
            probs.append(f"stored latent is {stored.shape[1]}-dim, expected 608")
        src_h, src_w = meta.get("source_hw", [0, 0])
        bucket = video_mod.pick_bucket(src_h, src_w) if src_h and src_w else None
        if bucket and bucket != meta.get("bucket"):
            probs.append(f"pick_bucket({src_h},{src_w}) = {bucket!r} != meta {meta.get('bucket')!r}")
        target_hw = tuple(video_mod.BUCKETS[bucket]) if bucket else None
        if target_hw and list(target_hw) != list(meta.get("target_hw", [])):
            probs.append(f"BUCKETS[{bucket}] = {target_hw} != meta target_hw {meta.get('target_hw')}")
        for p in probs:
            failures.append(f"{name}: {p}")

        n = pairs.shape[0]
        if args.max_pairs:
            n = min(n, args.max_pairs)
            pairs = pairs[:n]
            stored = stored[:n]

        # --- decode exactly the frames the pairs reference -----------------
        view = meta.get("view", "head")
        needed = set(int(x) for x in pairs.ravel())
        reader = video_mod.FrameReader(str(vpath), view)
        frames = read_frames(reader, needed)
        if target_hw:
            frames = {k: video_mod.resize_crop(v, target_hw) for k, v in frames.items()}

        imgs = np.stack([
            np.stack([frames[int(a)], frames[int(b)]]).astype(np.float32) / 255.0
            for a, b in pairs
        ])  # [n, 2, H, W, 3] in [0,1]

        outs = []
        for s in range(0, imgs.shape[0], args.batch_size):
            tok = encoder(imgs[s : s + args.batch_size], representation="ktoken_zcam")
            outs.append(tok.cpu().numpy().astype(np.float32))
        got = np.concatenate(outs, axis=0)

        # --- compare -------------------------------------------------------
        diff = np.abs(got - stored)
        denom = float(np.abs(stored).mean()) or 1.0
        rel = float(diff.mean()) / denom
        cos_rows = float(np.mean([cosine(got[i], stored[i]) for i in range(got.shape[0])]))
        slices = {"k_tokens[0:512]": K_SLICE, "z[512:576]": Z_SLICE, "camera[576:608]": CAM_SLICE}

        print(f"=== {name}")
        print(f"    video        : {vpath.name}  view={view}  source_hw={meta.get('source_hw')}")
        print(f"    geometry     : bucket={meta.get('bucket')} target_hw={meta.get('target_hw')} "
              f"(pick_bucket agrees: {bucket == meta.get('bucket')})")
        print(f"    pairs        : {got.shape[0]} of {pairs.shape[0]} encoded -> {got.shape[1]}-dim")
        print(f"    provenance   : checkpoint_sha256 {'MATCHES' if sha and sha == proto.get('checkpoint_sha256') else ('unchecked' if not sha else 'MISMATCH')} meta")
        print(f"    max |diff|   : {diff.max():.6g}")
        print(f"    mean |diff|  : {diff.mean():.6g}   (stored mean|.| = {denom:.6g})")
        print(f"    relative err : {rel:.4%}   (tolerance {args.rel_tol:.2%})")
        print(f"    mean cosine  : {cos_rows:.6f}   (tolerance {args.cos_tol})")
        for label, sl in slices.items():
            print(f"      {label:18s} max|diff|={diff[:, sl].max():.6g}  "
                  f"stored std={stored[:, sl].std():.6g}  got std={got[:, sl].std():.6g}")
        finite = bool(np.isfinite(got).all())
        print(f"    all finite   : {finite}")
        ok = (rel <= args.rel_tol and cos_rows >= args.cos_tol and finite and not probs)
        print(f"    -> {'PASS' if ok else 'FAIL'}\n")
        if not ok:
            if rel > args.rel_tol:
                failures.append(f"{name}: relative error {rel:.4%} > {args.rel_tol:.2%}")
            if cos_rows < args.cos_tol:
                failures.append(f"{name}: mean cosine {cos_rows:.6f} < {args.cos_tol}")
            if not finite:
                failures.append(f"{name}: non-finite values in re-encoded latents")

    print("=" * 68)
    if failures:
        print(f"LAM INFERENCE CHECK FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"LAM INFERENCE CHECK PASSED ({len(lat_dirs)} episode(s); latents reproduced "
          f"within {args.rel_tol:.2%} relative error and cosine >= {args.cos_tol})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
