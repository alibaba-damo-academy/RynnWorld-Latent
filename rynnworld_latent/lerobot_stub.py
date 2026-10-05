"""Inert ``lerobot`` stand-in for images that do not ship the real package.

cosmos's ``make_config()`` imports its own action-policy experiment registry, whose
dataset wrappers do module-level::

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames

That import runs for EVERY config load, including recipes that never touch DROID or
LeRobot data, so an image without lerobot cannot load any config at all — and an
offline image has no index to install it from. RynnWorld-Latent does not use
lerobot datasets, so inert stand-ins let the import succeed. The symbols are referenced only
at import time (type hints are deferred via ``from __future__ import annotations``) or
in code paths this repo never runs.

Each stand-in raises if instantiated, so a recipe that genuinely needs LeRobot data
fails loudly instead of silently running against a stub.

Shared by ``scripts/train.py`` (training) and ``scripts/inference/rollout.py``
(inference), so both present cosmos with the same import surface. Keeping one copy
matters: an entry point that loads configs under different imports than the
training run reports failures that cannot happen, and hides ones that can.
"""
from __future__ import annotations

import importlib.util
import sys
import types


def ensure() -> bool:
    """Install the stub when the real package is absent.

    Returns True if the stub was installed, False if a real ``lerobot`` is present
    (in which case nothing is shadowed).
    """
    if importlib.util.find_spec("lerobot") is not None:
        return False

    def _module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    lerobot = _module("lerobot")
    datasets = _module("lerobot.datasets")
    lerobot_dataset = _module("lerobot.datasets.lerobot_dataset")
    video_utils = _module("lerobot.datasets.video_utils")

    class LeRobotDataset:  # inert stand-in; never instantiated by this recipe
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("lerobot is stubbed out in this build")

    class LeRobotDatasetMetadata:  # inert stand-in; never instantiated here
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("lerobot is stubbed out in this build")

    def decode_video_frames(*args, **kwargs):
        raise NotImplementedError("lerobot is stubbed out in this build")

    lerobot_dataset.LeRobotDataset = LeRobotDataset
    lerobot_dataset.LeRobotDatasetMetadata = LeRobotDatasetMetadata
    video_utils.decode_video_frames = decode_video_frames
    datasets.lerobot_dataset = lerobot_dataset
    datasets.video_utils = video_utils
    lerobot.datasets = datasets
    return True
