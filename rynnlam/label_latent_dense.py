#!/usr/bin/env python3
"""Label each valid frame pair with the selected RynnLAM representation."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import threading
import uuid
import zipfile
from queue import Queue

import numpy as np

from rynnlam.inference import RynnLAMEncoder, file_sha256, implementation_sha256
from rynnlam.video import BUCKETS, FrameReader, pair_batches, resolve_source


def safe_name(value, *, nested=False):
    value = str(value)
    path = PurePosixPath(value)
    if (
        not value
        or value in {".", ".."}
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Invalid output identifier: {value!r}")
    if not nested and "/" in value:
        raise ValueError(f"Expected one path component: {value!r}")
    return value


def publish_file(local_path, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(local_path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(value, destination, scratch):
    local = Path(scratch) / f"{uuid.uuid4().hex}.json"
    try:
        local.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        publish_file(local, destination)
    finally:
        local.unlink(missing_ok=True)


def _check_array(archive, name, expected_shape, *, gap=None, pair_stride=1):
    with archive.open(name + ".npy") as member:
        version = np.lib.format.read_magic(member)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        if version not in readers:
            raise ValueError(f"Unsupported NPY version {version}")
        shape, fortran, dtype = readers[version](member)
        expected_dtype = np.dtype("float16") if gap is None else np.dtype("int32")
        if shape != expected_shape or fortran or dtype != expected_dtype:
            raise ValueError(f"Invalid {name} shape/dtype/order")
        row_bytes = shape[1] * dtype.itemsize
        rows_per_chunk = max(1, 1024 * 1024 // row_bytes)
        for first in range(0, shape[0], rows_per_chunk):
            count = min(rows_per_chunk, shape[0] - first)
            raw = member.read(count * row_bytes)
            if len(raw) != count * row_bytes:
                raise ValueError(f"Truncated {name}")
            values = np.frombuffer(raw, dtype=dtype).reshape(count, shape[1])
            if gap is None:
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite latent values")
            else:
                indices = np.arange(first, first + count) * pair_stride
                if not np.array_equal(values[:, 0], indices) or not np.array_equal(
                    values[:, 1], indices + gap
                ):
                    raise ValueError("Invalid frame-pair alignment")
        if member.read(1):
            raise ValueError(f"Unexpected bytes after {name}")


def output_metadata(path, protocol):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
    # Source-code hashes are provenance, not output semantics. Gating reuse on them
    # would force a full re-label after any unrelated code edit, so compare only the
    # fields that change the bytes we write. Bump schema_version if the geometry or
    # encoding itself changes.
    provenance = {"labeler_sha256", "implementation_sha256"}
    compare = tuple(sorted(k for k in protocol if k not in provenance))
    stored = meta["protocol"]
    if any(stored.get(k) != protocol[k] for k in compare):
        raise ValueError(
            "Existing output uses a different source/checkpoint/protocol; choose a new output directory"
        )
    n = int(meta["num_latents"])
    width = int(np.prod(meta["code_shape"]))
    stride, gap = protocol["pair_stride"], protocol["gap"]
    if n <= 0 or width <= 0 or stride < 1 or gap < 1:
        raise ValueError("Invalid latent shape or temporal stride")
    expected_count = max(0, (int(meta["num_frames"]) - 1 - gap) // stride + 1)
    if n != expected_count:
        raise ValueError(
            "Latent count does not match the video frame count and temporal stride"
        )
    with zipfile.ZipFile(path) as archive:
        _check_array(archive, "latent_action", (n, width))
        _check_array(archive, "pair_indices", (n, 2), gap=gap, pair_stride=stride)
    return meta


class _DecodeError:
    """Wrapper to distinguish decode errors from normal batch tuples in the queue."""

    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


def _prefetch_worker(pair_iter, queue, stop_event):
    """Background thread: decode batches and put them into the queue."""
    try:
        for batch in pair_iter:
            if stop_event.is_set():
                break
            queue.put(batch)
    except Exception as e:
        queue.put(_DecodeError(e))
    finally:
        queue.put(None)  # sentinel


def label_video(
    encoder,
    video,
    destination,
    protocol,
    *,
    view="head",
    batch_size=32,
    scratch=None,
    prefetch=2,
):
    reader = FrameReader(video, view)
    pair_iter = pair_batches(
        reader,
        gap=protocol["gap"],
        batch_size=batch_size,
        bucket=protocol["bucket"],
        start=protocol["start_frame"],
        end=protocol["end_frame"],
        pair_stride=protocol["pair_stride"],
    )
    queue = Queue(maxsize=prefetch)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_prefetch_worker, args=(pair_iter, queue, stop_event), daemon=True
    )
    thread.start()

    shape = bucket = native_hw = None
    count = 0
    try:
        # Keep long-video token arrays on local disk rather than accumulating in RAM.
        with tempfile.TemporaryDirectory(dir=scratch, prefix="rynnlam-label-") as work:
            work = Path(work)
            with (work / "latent.raw").open("wb") as latent_file, (
                work / "pairs.raw"
            ).open("wb") as pair_file:
                while True:
                    item = queue.get()
                    if item is None:
                        break
                    if isinstance(item, _DecodeError):
                        raise item.exc
                    images, pairs, bucket, native_hw = item
                    tokens = (
                        encoder(images, representation=protocol["representation"])
                        .cpu()
                        .numpy()
                    )
                    current_shape = list(tokens.shape[1:])
                    if shape is not None and current_shape != shape:
                        raise ValueError("Token shape changed within the video")
                    shape = current_shape
                    encoded = tokens.reshape(len(tokens), -1).astype(np.float16)
                    if not np.isfinite(encoded).all():
                        raise ValueError("Non-finite values after float16 conversion")
                    encoded.tofile(latent_file)
                    pairs.tofile(pair_file)
                    count += len(tokens)
            meta = {
                "protocol": protocol,
                "view": view,
                "num_latents": count,
                "num_frames": reader.num_frames,
                "unpaired_tail_frames": reader.num_frames
                - ((count - 1) * protocol["pair_stride"] + protocol["gap"] + 1),
                "code_shape": shape,
                "bucket": bucket,
                "target_hw": list(BUCKETS[bucket]),
                "source_hw": list(native_hw),
                "fps": reader.fps,
            }
            latent = np.memmap(
                work / "latent.raw",
                dtype=np.float16,
                mode="r",
                shape=(count, int(np.prod(shape))),
            )
            pairs = np.memmap(
                work / "pairs.raw", dtype=np.int32, mode="r", shape=(count, 2)
            )
            local = work / "latent.npz"
            try:
                # Uncompressed: zlib on dense float16 latents saves nothing
                # (output is exactly width*2 bytes per latent) but burns CPU on
                # every episode, which matters when decode already starves the GPU.
                np.savez(
                    local,
                    latent_action=latent,
                    pair_indices=pairs,
                    meta=json.dumps(meta),
                )
            finally:
                latent._mmap.close()
                pairs._mmap.close()
            output_metadata(local, protocol)
            publish_file(local, destination)
        return meta
    finally:
        stop_event.set()
        thread.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="One RGB video")
    source.add_argument(
        "--metadata", type=Path, help="JSON list of dataset/episode_id/views entries"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--representation",
        choices=["ktoken", "z", "zcam", "hints_pool", "ktoken_zcam"],
        default="ktoken",
    )
    parser.add_argument("--view", default="head", help="View name for --video")
    parser.add_argument(
        "--views", nargs="+", help="Only label these view names from --metadata"
    )
    parser.add_argument(
        "--gap", type=int, default=4, help="RGB frame interval within each pair"
    )
    parser.add_argument(
        "--pair-stride",
        "--stride",
        dest="pair_stride",
        type=int,
        default=4,
        help="Step between pair start frames; default 4 gives (0,4),(4,8),...",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Decode prefetch queue depth for pipeline overlap",
    )
    parser.add_argument("--bucket", choices=["auto", *BUCKETS], default="auto")
    parser.add_argument(
        "--normalization", choices=["imagenet", "none"], default="imagenet"
    )
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--scratch", type=Path, help="Local scratch directory, default system temp"
    )
    parser.add_argument(
        "--index-every",
        type=int,
        default=1000,
        help="Flush the episode index after this many episodes",
    )
    parser.add_argument(
        "--cursor-every",
        type=int,
        default=20,
        help="Flush the resume cursor after this many episodes; a restart resumes "
        "from the cursor instead of re-validating every finished label",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Validate existing labels and rebuild the index without inference",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow Python object deserialization; use only for a trusted checkpoint",
    )
    args = parser.parse_args(argv)
    if (
        args.gap < 1
        or args.pair_stride < 1
        or args.batch_size < 1
        or args.index_every < 1
        or args.cursor_every < 1
        or not 0 <= args.shard < args.num_shards
    ):
        parser.error(
            "Require gap/pair-stride/batch-size/index-every/cursor-every > 0 and 0 <= shard < num-shards"
        )
    if args.overwrite and args.rebuild_index:
        parser.error("--overwrite and --rebuild-index cannot be combined")
    if args.metadata:
        with args.metadata.open(encoding="utf-8") as handle:
            items = json.load(handle)
        base = args.metadata.resolve().parent
        if not isinstance(items, list):
            parser.error("metadata must contain a JSON list")
    else:
        items = [
            {
                "dataset": "videos",
                "episode_id": args.video.stem,
                "views": {args.view: {"video_path": str(args.video.resolve())}},
            }
        ]
        base = Path.cwd()
    identity = set()
    for item in items:
        key = (safe_name(item["dataset"]), safe_name(item["episode_id"], nested=True))
        if key in identity:
            parser.error(f"Duplicate dataset/episode_id: {key}")
        identity.add(key)
    items = items[args.shard :: args.num_shards]
    checkpoint_hash = file_sha256(args.checkpoint)
    implementation_hash = implementation_sha256()
    labeler_hash = file_sha256(__file__)
    if args.scratch:
        args.scratch.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resume state: a preemptible run skips the already-processed prefix instead of
    # re-validating every finished label (output_metadata reads both arrays, far too
    # slow to repeat over millions of episodes). Episodes that failed are NOT skipped
    # — a failed view means the episode is incomplete, so it is retried on every
    # restart, which also gives transient errors another chance.
    stem = args.metadata.stem if args.metadata else "videos"
    progress_path = (
        args.output_dir / "progress" / f"{stem}.shard{args.shard:04d}.json"
    )
    protocol_key = {
        "checkpoint_sha256": checkpoint_hash,
        "gap": args.gap,
        "pair_stride": args.pair_stride,
        "representation": args.representation,
        "normalization": args.normalization,
        "precision": args.precision,
        "bucket": args.bucket,
        "views": sorted(args.views) if args.views else None,
    }
    total_items = len(items)
    cursor, retry = 0, set()
    if not args.overwrite and not args.rebuild_index and progress_path.exists():
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        if state.get("protocol_key") == protocol_key:
            cursor = max(0, min(int(state.get("cursor", 0)), total_items))
            retry = {int(i) for i in state.get("failed_positions", [])}
        else:
            print(
                f"[resume] {stem}: protocol changed since the last run, restarting from 0",
                flush=True,
            )
    if cursor:
        print(
            f"[resume] {stem}: skipping {cursor}/{total_items} processed episodes, "
            f"retrying {len(retry)} failed ones",
            flush=True,
        )
    index, errors = [], []
    labeled = 0
    # The failure file is shared by every dataset of this shard, so keep the records
    # of other datasets and replace only this metadata file's own entries.
    failed_path = args.output_dir / f"failed_shard{args.shard:04d}.json"
    carried = []
    if not args.overwrite and failed_path.exists():
        try:
            previous = json.loads(failed_path.read_text(encoding="utf-8"))
        except Exception:
            previous = []
        if isinstance(previous, list):
            carried = [e for e in previous if e.get("metadata") != stem]
    encoder = None
    index_path = args.output_dir / f"index_shard{args.shard:04d}.json"
    with tempfile.TemporaryDirectory(
        dir=args.scratch, prefix="rynnlam-index-"
    ) as scratch:

        def flush_cursor(position, failing):
            save_json(
                {
                    "protocol_key": protocol_key,
                    "cursor": position,
                    "total": total_items,
                    "failed_positions": sorted(failing),
                },
                progress_path,
                scratch,
            )

        failing = set(retry)
        since_flush = 0
        for item_index, item in enumerate(items, start=1):
            if item_index <= cursor and item_index not in retry:
                continue
            since_flush += 1
            dataset = safe_name(item["dataset"])
            episode = safe_name(item["episode_id"], nested=True)
            views = {}
            item_failed = False
            for view, source_view in item["views"].items():
                safe_name(view)
                if args.views and view not in args.views:
                    continue
                destination = (
                    args.output_dir
                    / "latents"
                    / dataset
                    / episode
                    / view
                    / "latent.npz"
                )
                try:
                    video, source_size, source_mtime_ns = resolve_source(
                        source_view["video_path"], base
                    )
                    start = int(
                        source_view.get("start_frame", item.get("start_frame", 0)) or 0
                    )
                    end = source_view.get("end_frame", item.get("end_frame"))
                    protocol = {
                        "schema_version": 2,
                        "checkpoint_sha256": checkpoint_hash,
                        "implementation_sha256": implementation_hash,
                        "labeler_sha256": labeler_hash,
                        "source": str(video),
                        "source_size": source_size,
                        "source_mtime_ns": source_mtime_ns,
                        "gap": args.gap,
                        "pair_stride": args.pair_stride,
                        "start_frame": start,
                        "end_frame": int(end) if end is not None else None,
                        "bucket": args.bucket,
                        "normalization": args.normalization,
                        "representation": args.representation,
                        "precision": args.precision,
                    }
                    if destination.exists() and not args.overwrite:
                        meta = output_metadata(destination, protocol)
                    elif args.rebuild_index:
                        raise FileNotFoundError(f"Missing label {destination}")
                    else:
                        if encoder is None:
                            encoder = RynnLAMEncoder(
                                args.checkpoint,
                                device=args.device,
                                normalize=args.normalization == "imagenet",
                                precision=args.precision,
                                trust_checkpoint=args.trust_checkpoint,
                                checkpoint_sha256=checkpoint_hash,
                            )
                        meta = label_video(
                            encoder,
                            video,
                            destination,
                            protocol,
                            view=view,
                            batch_size=args.batch_size,
                            scratch=args.scratch,
                            prefetch=args.prefetch,
                        )
                    views[view] = {
                        "video_path": str(video),
                        "latent_path": str(destination.resolve()),
                        "num_frames": meta["num_frames"],
                        "num_latents": meta["num_latents"],
                        "unpaired_tail_frames": meta["unpaired_tail_frames"],
                        "code_shape": meta["code_shape"],
                        "fps": meta["fps"],
                        "bucket": meta["bucket"],
                    }
                    print(
                        f"{dataset}/{episode}/{view}: {meta['num_latents']} latents, shape={meta['code_shape']}",
                        flush=True,
                    )
                except Exception as error:
                    errors.append(
                        {
                            "metadata": stem,
                            "dataset": dataset,
                            "episode_id": episode,
                            "view": view,
                            "error": str(error),
                        }
                    )
                    item_failed = True
                    print(f"FAILED {dataset}/{episode}/{view}: {error}", flush=True)
            if item_failed:
                failing.add(item_index)
            else:
                failing.discard(item_index)
            if views:
                labeled += 1
                index.append(
                    {
                        "dataset": dataset,
                        "episode_id": episode,
                        "caption": item.get("caption", ""),
                        "gap": args.gap,
                        "pair_stride": args.pair_stride,
                        "representation": args.representation,
                        "views": views,
                    }
                )
            if item_index % args.index_every == 0:
                save_json(index, index_path, scratch)
            if since_flush >= args.cursor_every:
                since_flush = 0
                flush_cursor(item_index, failing)
                save_json(carried + errors, failed_path, scratch)
        flush_cursor(total_items, failing)
        save_json(index, index_path, scratch)
        save_json(carried + errors, failed_path, scratch)
        # No done marker while failures remain: the next run retries exactly those
        # positions (they fail fast) instead of skipping the dataset for good.
        if not errors:
            save_json(
                {
                    "metadata": stem,
                    "protocol_key": protocol_key,
                    "total": total_items,
                    "labeled": labeled,
                    "failed": 0,
                },
                args.output_dir / "done" / f"{stem}.shard{args.shard:04d}.json",
                scratch,
            )
    print(
        f"Indexed {len(index)} episodes this run; labeled={labeled} "
        f"failed={len(errors)} of {total_items} -> {index_path}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
