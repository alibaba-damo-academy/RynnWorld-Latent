#!/usr/bin/env python3
"""Cross-action transfer visualization.

For each (A, B) pair (same dataset, different episodes):
  * gen_self  = generate from A's first frame + A's own action  -> saved as gen|GT(A)
  * gen_cross = generate from A's first frame + B's latent action -> saved as A|B|gen

If action-following works, gen_cross should move like B while keeping A's content/style.
Usage mirrors scripts/inference/rollout.py (same env / checkpoint loading); export the
same RYNNWORLD_ACTION_* film switches used in training, or the model structure will not
match the checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rynnworld_latent.inference as er
from _video_utils import hstack_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--staged-root", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default="rynnworld_latent_edge_manifest_fullft")
    ap.add_argument("--pairs", required=True, help="A:B,A:B,... dataset indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()
    er.resolve_arg_paths(args, "manifest_dir", "staged_root", "checkpoint", "out")

    device = "cuda"
    ds = er.build_rollout_dataset(args)
    model = er.load_world_model(args)
    out = Path(args.out); (out / "self").mkdir(parents=True, exist_ok=True); (out / "cross").mkdir(parents=True, exist_ok=True)
    from cosmos_framework.tools.visualize.video import save_img_or_video

    pairs = [tuple(map(int, p.split(":"))) for p in args.pairs.split(",")]
    for a_idx, b_idx in pairs:
        sa = ds._load_one(a_idx)
        sb = ds._load_one(b_idx)
        fps = int(sa["conditioning_fps"])

        # --- gen_self: A frame0 + A action ---
        batch_a, gt_a = er.build_rollout_batch(model, sa, device)
        with torch.no_grad():
            o = model.generate_samples_from_batch(batch_a, seed=[args.seed], num_steps=args.num_steps, guidance=args.guidance)
            gen_self = model.decode(o["vision"][0])
        gen_self01 = er.pm1_to_unit(gen_self)
        gt_a01 = er.pm1_to_unit(gt_a)
        g, t = er.crop_common(gen_self01, gt_a01)
        save_img_or_video(g.squeeze(0), str(out / "self" / f"gen_{a_idx:07d}"), fps=fps)
        save_img_or_video(t.squeeze(0), str(out / "self" / f"gt_{a_idx:07d}"), fps=fps)

        # --- gen_cross: A frame0 + B action ---
        batch_x, _ = er.build_rollout_batch(model, sa, device)
        batch_x["action"] = [sb["action"].to(torch.float32).to(device)]
        with torch.no_grad():
            ox = model.generate_samples_from_batch(batch_x, seed=[args.seed], num_steps=args.num_steps, guidance=args.guidance)
            gen_cross = model.decode(ox["vision"][0])
        gen_cross01 = er.pm1_to_unit(gen_cross)
        gx, _ = er.crop_common(gen_cross01, gt_a01)

        # write the three panels then hstack: A_gt | B_gt | gen_cross
        gt_b = er.build_rollout_batch(model, sb, device)[1]
        gt_b01 = er.pm1_to_unit(gt_b)
        _, tb = er.crop_common(gt_b01, gt_b01)
        pa = out / "cross" / f"_a_{a_idx:07d}.mp4"
        pb = out / "cross" / f"_b_{b_idx:07d}.mp4"
        pg = out / "cross" / f"_g_{a_idx:07d}x{b_idx:07d}.mp4"
        save_img_or_video(t.squeeze(0), str(pa.with_suffix("")), fps=fps)
        save_img_or_video(tb.squeeze(0), str(pb.with_suffix("")), fps=fps)
        save_img_or_video(gx.squeeze(0), str(pg.with_suffix("")), fps=fps)
        hstack_videos([pa, pb, pg], str(out / "cross" / f"xfer_{a_idx:07d}x{b_idx:07d}.mp4"))
        for f in (pa, pb, pg):
            f.unlink(missing_ok=True)
        print(f"[xfer] A={a_idx} B={b_idx} done", flush=True)

    print("CROSS TRANSFER DONE")


if __name__ == "__main__":
    main()
