"""``rynnworld_latent_edge_*`` — Cosmos3-Edge RynnWorld-Latent SFT / post-train recipes.

Trains a forward-dynamics world model conditioned on RynnLAM latent actions
(608-dim ``ktoken_zcam``: k_tokens 512 | z 64 | camera 32) on the Cosmos3-Edge
(Nemotron-2B) backbone.

All recipes read chunk records from ``MANIFEST_DIR`` and pixels from ``STAGED_ROOT``
(or the source videos in place when it is empty):

* ``rynnworld_latent_edge_manifest`` — generation branch + action heads, via the
  ``optimizer.keys_to_select`` allowlist. Warm-starts from the Cosmos3-Edge base.
* ``rynnworld_latent_edge_manifest_fullft`` — full-parameter fine-tune of the entire
  3.37B net, fps pinned to 10.0. Warm-starts from the Cosmos3-Edge base.
* ``rynnworld_latent_edge_manifest_v3_fullft`` — same as ``_fullft`` but ``fps=None``
  (per-record real fps drives the mRoPE time step). The released weights are a v3
  checkpoint; see that node's comment for the fps caveat before using it.
* ``rynnworld_latent_edge_posttrain`` — downstream embodiment post-training: warm-starts
  from a TRAINED RynnWorld-Latent film checkpoint (not the base) and fine-tunes on your
  own manifest, preserving the trained action heads (only ``net_ema.`` is skipped).

Usage (1 node, 8 GPU)::

    MANIFEST_DIR=/path/to/manifest \
    STAGED_ROOT=/path/to/staged480 \
    BASE_CHECKPOINT_PATH=<Cosmos3-Edge DCP dir> \
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \
    IMAGINAIRE_OUTPUT_ROOT=./outputs \
    torchrun --nproc_per_node=8 scripts/train.py \
        --sft-toml configs/train/edge_fullft.toml

Downstream post-train (warm-start from a trained film checkpoint, your own embodiment).
The released film weights are safetensors; ``scripts/checkpoints/convert_released_to_dcp.py``
re-serializes them to the DCP dir ``BASE_CHECKPOINT_PATH`` expects (CPU-only, no training)::

    MANIFEST_DIR=/path/to/your/embodiment/manifest \
    BASE_CHECKPOINT_PATH=<trained film DCP dir, containing model/> \
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \
    IMAGINAIRE_OUTPUT_ROOT=./outputs \
    torchrun --nproc_per_node=8 scripts/train.py \
        --sft-toml configs/posttrain/downstream.toml
"""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from rynnworld_latent.dataset import ACTION_DIM
from rynnworld_latent.sft_dataset import get_rynnworld_manifest_sft_dataset

cs = ConfigStore.instance()

# Text-free mode: drop ALL text conditioning so the future video is explained only
# by the first frame + latent action. cfg_dropout_rate=1.0 empties the caption and
# the append_*/json toggles stop the metadata augmentors from re-injecting text,
# leaving the text branch as just BOS/EOS. Off by default.
_TEXT_FREE = os.environ.get("RYNNWORLD_TEXT_FREE", "0").strip() == "1"

_edge_base = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            # Local checkpoint SKU: object store disabled, no bucket/credentials, no
            # S3/GCS backend. RynnWorld-Latent reads/writes checkpoints on the local
            # filesystem only -- it must never reach a cloud object-store API.
            {"override /checkpoint": "local"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="rynnworld_latent_sft",
            name="rynnworld_latent_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(EDGE_MODEL_CONFIG),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=2.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            # Never monkeypatch boto3 for GCS: RynnWorld-Latent checkpoints stay on
            # the local filesystem and must not activate any cloud object-store path.
            enable_gcs_patch_in_boto3=False,
            keys_not_to_resume=[],
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="rynnworld_latent",
            max_samples_per_batch=128,
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    rynnworld_latent=dict(
                        ratio=1,
                        dataset=L(get_rynnworld_manifest_sft_dataset)(
                            manifest_dir="${oc.env:MANIFEST_DIR}",
                            # STAGED_ROOT is optional: empty means "read the original
                            # source videos in place" (no 480p staging copy). Default it
                            # so a bare `train.sh` / direct invocation doesn't need it set.
                            staged_root="${oc.env:STAGED_ROOT,''}",
                            fps=10.0,
                            mode="forward_dynamics",
                            action_normalization="quantile",
                            viewpoint="ego_view",
                            resolution="480",
                            max_action_dim=ACTION_DIM,
                            cfg_dropout_rate=1.0 if _TEXT_FREE else 0.0,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            format_prompt_as_json=not _TEXT_FREE,
                            append_viewpoint_info=not _TEXT_FREE,
                            append_duration_fps_timestamps=not _TEXT_FREE,
                            append_resolution_info=not _TEXT_FREE,
                            append_idle_frames=False,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

_edge_base["model"]["config"]["max_action_dim"] = ACTION_DIM

# VAE encode durations (pixel frames). 81 = the standard 20-action chunk (4*20+1).
# A manifest may also carry whole-view short chunks (na = L < 20), giving durations
# 4n+1 for n in [K, 19]; register the full range so the tokenizer encodes them at
# exact length. Unregistered durations stay CORRECT (the VAE pads to the temporal
# window then trims the latent back, wan2pt2_vae_4x16x16.py) but waste encode
# compute. RYNNWORLD_MIN_CHUNK_LATENTS must be <= the smallest `na` in the manifest.
_MIN_CHUNK_LATENTS = max(1, min(20, int(os.environ.get("RYNNWORLD_MIN_CHUNK_LATENTS", "20"))))
_ENCODE_DURATIONS = [4 * n + 1 for n in range(_MIN_CHUNK_LATENTS, 21)]

# A chunk is 20 latents -> 81 observation frames; pin the VAE encode durations.
_edge_base["model"]["config"]["tokenizer"]["encode_exact_durations"] = _ENCODE_DURATIONS

# The stock tokenizer config's vae_path is relative and only resolves from the
# cosmos-framework root, where no weights ship. WAN_VAE_PATH is a documented
# requirement of every entry point, so prefer it and keep the relative path only
# as a fallback for images that vendor the weights in-tree.
_edge_base["model"]["config"]["tokenizer"]["vae_path"] = (
    "${oc.env:WAN_VAE_PATH,pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth}"
)

# Uncap packed-sequence length for full vision context
_edge_base["model"]["config"]["max_num_tokens_after_packing"] = -1

# Vision flow-matching loss weight (balanced against action_loss_weight=10)
_edge_base["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


# ---------------------------------------------------------------------------
# Generation-branch recipe: the optimizer.keys_to_select allowlist above trains
# only the generation branch plus the action heads. The data path is
# manifest-driven rather than episode-directory based because at ~616k chunks,
# materializing one mp4 per chunk would cost ~1.8 TB and ~1.9M files.
# ---------------------------------------------------------------------------
rynnworld_latent_edge_manifest = copy.deepcopy(_edge_base)
rynnworld_latent_edge_manifest["job"]["name"] = "rynnworld_latent_edge_manifest"

cs.store(
    group="experiment",
    package="_global_",
    name="rynnworld_latent_edge_manifest",
    node=rynnworld_latent_edge_manifest,
)


# ---------------------------------------------------------------------------
# Full-parameter fine-tuning variant: trains the ENTIRE 3.37B net (understanding
# backbone + generation branch + action heads), not just the gen branch.
#
# ``keys_to_select=[]`` disables the allowlist in optimizer.py, so no parameter is
# frozen. This is aggressive: NVIDIA's own action recipes freeze the understanding
# stream precisely to protect the pretrained VLM priors, so full-FT risks
# catastrophic forgetting. For a quality run, give the backbone a ~10-100x lower
# LR; the action heads keep their 5x multiplier so the zero-initialised pathway
# still adapts fast.
# ---------------------------------------------------------------------------
rynnworld_latent_edge_manifest_fullft = copy.deepcopy(rynnworld_latent_edge_manifest)
rynnworld_latent_edge_manifest_fullft["job"]["name"] = "rynnworld_latent_edge_manifest_fullft"
rynnworld_latent_edge_manifest_fullft["optimizer"]["keys_to_select"] = []

cs.store(
    group="experiment",
    package="_global_",
    name="rynnworld_latent_edge_manifest_fullft",
    node=rynnworld_latent_edge_manifest_fullft,
)


# ---------------------------------------------------------------------------
# Full-corpus (v3) recipe: identical to edge_manifest_fullft except fps=None.
#
# fps=None makes manifest_dataset.py fall through to each record's own probed fps
# (``self._fps_override if not None else the record's fps``) instead of overriding
# every sample with 10.0. That value is the mRoPE temporal position step's
# denominator (base_fps/fps = 24/fps, via diffusion_expert_config.enable_fps_modulation),
# so switching 10.0 -> per-record fps globally rescales the time position encoding (on a
# mixed-fps corpus, roughly 2.7x on the corpus-mean step). Two consequences:
#   * A checkpoint trained at fps=10.0 CANNOT be warm-started into this recipe -- the
#     time encoding would be misaligned. v3 trains from the Cosmos3-Edge base.
#   * YOUR manifest MUST carry a real per-record ``fps`` field. A wrong/placeholder fps
#     SILENTLY misaligns the mRoPE timestep (finite loss, no error); a missing one raises
#     at record load. Only pin fps=10.0 (the _fullft recipe) if your corpus truly is
#     uniform 10 fps.
# The released RynnWorld-Latent weights are a v3 (fps=None) checkpoint; this recipe
# reproduces that training configuration.
# ---------------------------------------------------------------------------
rynnworld_latent_edge_manifest_v3_fullft = copy.deepcopy(rynnworld_latent_edge_manifest_fullft)
rynnworld_latent_edge_manifest_v3_fullft["job"]["name"] = "rynnworld_latent_edge_manifest_v3_fullft"
rynnworld_latent_edge_manifest_v3_fullft["dataloader_train"]["dataloader"]["datasets"]["rynnworld_latent"]["dataset"]["fps"] = None

cs.store(
    group="experiment",
    package="_global_",
    name="rynnworld_latent_edge_manifest_v3_fullft",
    node=rynnworld_latent_edge_manifest_v3_fullft,
)


# ---------------------------------------------------------------------------
# Downstream embodiment post-training: initialize from a TRAINED RynnWorld-Latent
# film checkpoint (BASE_CHECKPOINT_PATH -> the DCP dir containing model/), not the
# Cosmos3-Edge base, then fine-tune on your own embodiment's manifest (MANIFEST_DIR).
#
# The base recipes set keys_to_skip_loading to also drop action2llm / llm2action /
# action_modality_embed / action_pos_embed -- correct when warm-starting from the base
# Edge, where those action params do not exist yet and are reseeded from scratch. A
# downstream post-train warm-starts from a checkpoint that ALREADY has trained action
# heads, so the skip list is narrowed to ["net_ema."] only: the film action pathway is
# PRESERVED and the EMA re-warms from the loaded regular weights. load_training_state
# stays False (inherited), so the optimizer/scheduler/iteration start fresh while the
# weights carry over. Data-agnostic: the embodiment is chosen purely by MANIFEST_DIR at
# launch -- nothing here names a dataset. See configs/posttrain/downstream.toml.
#
# Derives from the v3 (fps=None) recipe, NOT _fullft (fps=10.0). The released
# RynnWorld-Latent weights are a v3 checkpoint and a downstream manifest carries each
# record's real fps, so pinning 10.0 here would rescale the mRoPE timestep (24/fps) by
# ~3x and silently misalign against the warm-started weights -- the exact caveat
# documented on _v3_fullft above. If you instead warm-start from your OWN fps=10.0
# stage-1 checkpoint, override dataset.fps=10.0 at launch.
# ---------------------------------------------------------------------------
rynnworld_latent_edge_posttrain = copy.deepcopy(rynnworld_latent_edge_manifest_v3_fullft)
rynnworld_latent_edge_posttrain["job"]["name"] = "rynnworld_latent_edge_posttrain"
rynnworld_latent_edge_posttrain["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]

cs.store(
    group="experiment",
    package="_global_",
    name="rynnworld_latent_edge_posttrain",
    node=rynnworld_latent_edge_posttrain,
)
