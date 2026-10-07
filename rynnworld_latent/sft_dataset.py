"""SFT dataset factory: manifest dataset + ActionTransformPipeline.

Defines ActionSFTDataset/ActionIterableShuffleDataset locally to avoid
pulling in the lerobot dependency from the cosmos datasets package.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

from rynnworld_latent.dataset import ACTION_DIM


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ActionTransformPipeline per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view with per-epoch index shuffling across ranks/workers."""

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:
        return len(self._dataset)

    def _resume_offset(self) -> int:
        """Committed trainer iteration at this run's start, from the DCP marker.

        A resume is a fresh torchrun, so ``epoch`` alone restarts at 0 and would
        re-draw the same permutation prefix as the original run -- a restart would
        train on the same leading records again instead of continuing. Keying the
        seed on the committed iteration makes each run's stream a distinct
        deterministic function of training progress, so a resume picks up the data
        order where it left off. The marker advances at checkpoint-save granularity.
        """
        import glob

        out = os.environ.get("IMAGINAIRE_OUTPUT_ROOT")
        if not out:
            return 0
        best = 0
        for p in glob.glob(os.path.join(out, "*", "*", "*", "checkpoints", "latest_checkpoint.txt")):
            try:
                with open(p) as fh:
                    best = max(best, int("".join(ch for ch in fh.read() if ch.isdigit()) or 0))
            except OSError:
                continue
        return best

    def __iter__(self):
        total = len(self._dataset)
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        base = self._seed + self._resume_offset()
        if global_shard == 0:
            print(f"[sft_dataset] permutation seed base={base} (seed={self._seed} + committed iter)")
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(base + epoch)
            order = torch.randperm(total, generator=g).tolist()
            shard = order[global_shard::total_shards]
            if not shard:
                # Dataset smaller than the shard count (total < world_size*num_workers),
                # e.g. a 1-sample quickstart bundle: this worker's strided slice is empty,
                # so the `while True` loop would spin forever yielding nothing and stall
                # the DataLoader the moment it waits on this worker. Fall back to the full
                # order so every shard produces data. No-op once total >= total_shards.
                shard = order
            for idx in shard:
                yield self._dataset[idx]
            epoch += 1


def get_rynnworld_manifest_sft_dataset(
    *,
    manifest_dir: str,
    staged_root: str | None = None,
    fps: float | None = None,
    mode: str = "forward_dynamics",
    action_normalization: str | None = "quantile",
    viewpoint: str = "ego_view",
    stats_path: str | None = None,
    limit: int = 0,
    src_remap: str | None = None,
    resolution: str | int = "480",
    max_action_dim: int = ACTION_DIM,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the manifest-driven RynnWorld-Latent SFT dataset."""
    from rynnworld_latent.manifest_dataset import RynnWorldManifestDataset

    dataset = RynnWorldManifestDataset(
        manifest_dir=manifest_dir,
        staged_root=staged_root,
        fps=fps,
        mode=mode,
        action_normalization=action_normalization,
        viewpoint=viewpoint,
        stats_path=stats_path,
        limit=limit,
        src_remap=src_remap,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
