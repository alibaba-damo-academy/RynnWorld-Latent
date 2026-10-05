"""Shared video-compositing helpers for the example scripts."""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path


def hstack_videos(paths, out_path, height: int = 480) -> None:
    """Composite N videos side by side, each scaled to a common ``height``.

    ``scale=-2:H`` keeps every panel's aspect ratio (width auto-rounded to an even
    number), so panels may end up different widths but are never stretched or
    letterboxed.

    ffmpeg writes to a node-local temp file which is then copied to ``out_path``:
    some network / object-store FUSE mounts reject the append/seek pattern ffmpeg
    uses when writing straight to them. The temp name is unique per call (pid +
    uuid) so parallel processes compositing same-named outputs never race.
    """
    paths = [str(p) for p in paths]
    out_path = str(out_path)
    n = len(paths)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    scale = "".join(f"[{i}:v]scale=-2:{height}[v{i}];" for i in range(n))
    tmp = str(Path("/tmp") / f"{os.getpid()}_{uuid.uuid4().hex}_{Path(out_path).name}")
    cmd += [
        "-filter_complex",
        scale + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}",
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        tmp,
    ]
    subprocess.run(cmd, check=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp, out_path)
    Path(tmp).unlink(missing_ok=True)
