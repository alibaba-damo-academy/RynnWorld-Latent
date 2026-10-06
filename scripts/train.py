#!/usr/bin/env python3
"""Training entrypoint for RynnWorld-Latent SFT on the Cosmos3-Edge backbone.

cosmos_framework's ``make_config()`` only imports its own shipped experiments,
so the external ``rynnworld_latent_edge_manifest*`` LazyDicts would never reach Hydra's
ConfigStore if we launched ``-m cosmos_framework.scripts.train`` directly.

Importing ``rynnworld_latent.experiment_config`` has the side effect of calling
``ConfigStore.instance().store(group="experiment", name="rynnworld_latent_edge_manifest",
...)`` for each recipe. We do that first, then hand off to the stock cosmos train
script (run as ``__main__`` with the same argv) so ``load_experiment_from_toml``
can resolve ``experiment=rynnworld_latent_edge_manifest_fullft``.

Launch (matches scripts/train.sh)::

    torchrun ... scripts/train.py --sft-toml configs/train/edge_fullft.toml
"""

import os
import runpy
import sys
import types
from pathlib import Path

# Help must not import the model stack, install patches, or initialize training.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Additional arguments and overrides are forwarded to cosmos_framework.scripts.train.",
    )
    parser.add_argument("--sft-toml", metavar="PATH", help="Training recipe TOML, e.g. configs/train/edge_fullft.toml")
    parser.print_help()
    raise SystemExit(0)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("COSMOS_ROOT", str(_REPO_ROOT / "third_party" / "cosmos-framework")))
sys.path.insert(0, str(_REPO_ROOT))

# Offline av compatibility shim. Some images ship av>=16, which removed
# the ``av.option`` submodule, but lerobot's pyav_utils references
# ``av.option.Option`` in a module-level type annotation -> ImportError on
# ``import lerobot``. (``av.codec`` still exists in av>=16, so only ``av.option``
# needs the stub.) Defensive: only relevant if a real lerobot is present.
try:
    import av as _av
except Exception:  # noqa: BLE001 - av may be absent entirely
    _av = None
if _av is not None and not hasattr(_av, "option"):
    _av_option = types.ModuleType("av.option")

    class _AvOptionStub:  # minimal stand-in for type annotations only
        pass

    _av_option.Option = _AvOptionStub
    _av.option = _av_option
    sys.modules["av.option"] = _av_option


def _ensure_lerobot_stub() -> None:
    """Provide an inert ``lerobot`` stub when the real package is absent.

    cosmos's ``make_config()`` imports its shipped action-policy experiments,
    whose dataset wrappers do module-level::

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import decode_video_frames

    An offline image has no lerobot and no package index to install it from. RynnWorld-Latent
    does not use lerobot datasets, so inert stand-ins let those imports succeed.

    The implementation lives in ``rynnworld_latent.lerobot_stub`` so
    ``scripts/inference/rollout.py`` installs the identical stub. Two copies would
    drift, and an entry point loading configs under different imports than the
    training run reports failures that cannot happen while hiding ones that can.
    """
    from rynnworld_latent.lerobot_stub import ensure

    ensure()


_ensure_lerobot_stub()


def _patch_sched_setaffinity_compat() -> None:
    """Make ``os.sched_setaffinity`` non-fatal.

    cosmos ``distributed.init()`` pins CPU affinity for NUMA placement but only
    catches ``pynvml.NVMLError``; under a constrained cpuset (e.g. when another
    job pins CPUs on this host) ``os.sched_setaffinity`` raises
    ``OSError: [Errno 22] Invalid argument`` and aborts startup. Affinity is an
    optimization, not a requirement, so swallow the error and continue.
    """
    original = os.sched_setaffinity

    def _setaffinity_compat(pid, cpus):
        try:
            return original(pid, cpus)
        except OSError as e:  # noqa: BLE001 - EINVAL under restricted cpuset
            print(f"[train.py] WARNING: sched_setaffinity failed ({e}); continuing unpinned")

    os.sched_setaffinity = _setaffinity_compat


_patch_sched_setaffinity_compat()


from rynnworld_latent.attention_sdpa_fallback import patch_sdpa_attention_backend

patch_sdpa_attention_backend()


def _patch_processor_local() -> None:
    """Load the Cosmos3-Edge VLM tokenizer from local files, not HF Hub.

    ``build_processor_lazy`` (invoked while building the model) resolves the
    tokenizer via ``CheckpointDirHf(repository="nvidia/Cosmos3-Edge").download()``,
    which shells out to ``uvx ... hf download`` — unavailable offline. We:
      1. patch ``CheckpointDirHf._download`` to return the local Edge dir for that
         repository (skips the download), and
      2. patch ``processors.build_processor`` to construct the processor from the
         local ``text_tokenizer/`` files with ``PreTrainedTokenizerFast``
         (``AutoProcessor`` cannot parse the Edge layout). ``build_processor_lazy``
         looks up ``build_processor`` fresh at call time, so this patch is honored.
    """
    edge_dir = os.environ.get("RYNNWORLD_EDGE_HF_DIR", "") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "cosmos_tokenizers", "edge",
    )
    tok_dir = os.path.join(edge_dir, "text_tokenizer")
    tok_file = os.path.join(tok_dir, "tokenizer.json")
    if not os.path.isfile(tok_file):
        print(f"[train.py] WARNING: local tokenizer missing at {tok_file}; processor patch skipped")
        return

    import types

    import cosmos_framework.data.generator.processors as _proc
    import cosmos_framework.utils.checkpoint_db as _ckptdb
    from transformers import PreTrainedTokenizerFast

    class _FakeProcessor:
        """Minimal stand-in for the cosmos Edge processor, backed by the local
        tokenizer. Implements the subset of the processor interface used by the
        training data pipeline (TextTokenizerTransform.tokenize_text) and model.
        """

        def __init__(self):
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tok_file, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
            )
            chat_tpl = os.path.join(tok_dir, "chat_template.jinja")
            if os.path.isfile(chat_tpl):
                with open(chat_tpl) as fh:
                    self.tokenizer.chat_template = fh.read()
            # Some code paths access proc.processor.tokenizer.
            self.processor = types.SimpleNamespace(tokenizer=self.tokenizer)

        @property
        def eos_id(self):
            return self.tokenizer.eos_token_id

        def tokenize_text(self, caption, is_video=False, use_system_prompt=False, system_prompt=None):
            from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption

            return tokenize_caption(
                caption,
                self.tokenizer,
                is_video=is_video,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )

        def encode(self, *args, **kwargs):
            return self.tokenizer.encode(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.tokenizer.decode(*args, **kwargs)

    _orig_download = _ckptdb.CheckpointDirHf._download

    def _local_download(self):
        if getattr(self, "repository", "") == "nvidia/Cosmos3-Edge":
            return edge_dir
        return _orig_download(self)

    _ckptdb.CheckpointDirHf._download = _local_download

    def _local_build_processor(local_path=None, *args, **kwargs):
        return _FakeProcessor()

    _proc.build_processor = _local_build_processor
    print(f"[train.py] processor patched -> local tokenizer: {tok_file}")


_patch_processor_local()

# Register the rynnworld_latent_edge_manifest* experiments into Hydra's ConfigStore.
import rynnworld_latent.experiment_config  # noqa: F401

# The reasoner config (e.g. 'cosmos_framework/model/generator/reasoner/.../
# Nemotron-2B-Dense-VL.json') is referenced relative to the cosmos-framework repo
# root, but torchrun launches us from the project dir. chdir to the cosmos root
# (as the model-build path expects) so those relative paths resolve. Imports and
# the data/checkpoint/output paths are absolute, so changing cwd is safe.
import cosmos_framework  # noqa: E402

_cosmos_root = os.path.dirname(os.path.dirname(os.path.abspath(cosmos_framework.__file__)))
os.chdir(_cosmos_root)
print(f"[train.py] cwd -> cosmos root: {_cosmos_root}")


def _patch_fully_shard_compat() -> None:
    """Drop the 'ignored_params' kwarg if this torch's fully_shard lacks it.

    cosmos's parallelize_vfm_network.py calls
    ``fully_shard(module=..., mesh=..., ignored_params=set())``, but an older
    torch's ``fully_shard`` has no ``ignored_params`` parameter. cosmos always passes an *empty* set, so stripping the kwarg is a
    no-op. Patch both the public and internal fully_shard before the parallelize
    modules import them (they do ``from torch.distributed.fsdp import
    fully_shard`` during model build, i.e. inside runpy below).
    """
    import inspect

    import torch.distributed.fsdp as _torch_fsdp

    original = _torch_fsdp.fully_shard
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        params = {}
    if "ignored_params" in params:
        return  # this torch supports it; nothing to do

    def _fully_shard_compat(module, *args, **kwargs):
        ignored = kwargs.pop("ignored_params", None)
        if ignored:
            raise NotImplementedError(
                "fully_shard got a non-empty ignored_params, but this torch build "
                "does not support the ignored_params argument."
            )
        return original(module, *args, **kwargs)

    _torch_fsdp.fully_shard = _fully_shard_compat
    try:
        import torch.distributed._composable.fsdp as _composable_fsdp

        _composable_fsdp.fully_shard = _fully_shard_compat
    except Exception:  # noqa: BLE001 - internal path may not exist; public patch suffices
        pass
    print("[train.py] patched fully_shard: stripped unsupported 'ignored_params' kwarg")


_patch_fully_shard_compat()


def _patch_init_weights_compat() -> None:
    """Tolerate old-torch DTensor init ops in the reasoner's _init_weights.

    On an older torch, an FSDP-sharded (DTensor) embedding has no
    sharding rule for ``aten.select.int``, so transformers'
    ``PreTrainedModel._init_weights`` raises NotImplementedError when it zeros
    ``module.weight.data[padding_idx]``. The checkpoint load overwrites these
    weights regardless, so swallowing that error during init is safe.
    """
    try:
        from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
            Nemotron3DenseVLPreTrainedModel,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[train.py] skipping _init_weights patch (import failed: {e})")
        return

    original = Nemotron3DenseVLPreTrainedModel._init_weights

    def _init_weights_compat(self, module, buffer_device=None):
        try:
            original(self, module, buffer_device)
        except NotImplementedError:
            # Old torch lacks a DTensor sharding rule for an init op (e.g.
            # aten.select.int). Checkpoint load overwrites these weights.
            pass

    Nemotron3DenseVLPreTrainedModel._init_weights = _init_weights_compat
    print("[train.py] patched Nemotron _init_weights to tolerate old-torch DTensor init ops")


_patch_init_weights_compat()


# ===========================================================================
# Action-conditioning weight initialization.
#
# Cosmos3-Edge ships a *trained* action pathway (action2llm / llm2action /
# action_modality_embed; pretrained action_dim=64, 32 embodiment domains). None of
# it transfers: those domain rows encode a 64-dim robot action space, which has no
# correspondence in RynnLAM's 608-dim [k_tokens 512 | z 64 | camera 32] latent. So
# the action layers are zero-initialised — an exact no-op at step 0 — and learned
# from scratch, leaving the rest of the pretrained backbone untouched.
#
# In forward_dynamics every action is conditioning, so llm2action runs a dummy
# path and does NOT affect training; only action2llm (encode) matters. Action
# layers are in keys_to_skip_loading, so the values written in init_weights
# survive the DCP main load.
# ===========================================================================


def _zero_action_pathway(self) -> None:
    """Zero the action layers so they start as a no-op (learned from scratch)."""
    import torch

    a2l = getattr(self, "action2llm", None)
    if a2l is not None:
        if hasattr(getattr(a2l, "fc", None), "weight"):
            torch.nn.init.zeros_(a2l.fc.weight)
        if hasattr(getattr(a2l, "bias", None), "weight"):
            torch.nn.init.zeros_(a2l.bias.weight)
    me = getattr(self, "action_modality_embed", None)
    if me is not None:
        torch.nn.init.zeros_(me)
    print("[train.py] action init: zero (learn from scratch)")


def _patch_action_init() -> None:
    """Register the action zero-init weight patch before the runpy hand-off."""
    try:
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    except Exception as e:  # noqa: BLE001
        print(f"[train.py] skipping action-init patch (import failed: {e})")
        return

    original_init_weights = Cosmos3VFMNetwork.init_weights

    def _init_weights_zero_action(self, buffer_device=None):
        original_init_weights(self, buffer_device)
        if not getattr(self.config, "action_gen", False):
            return
        try:
            _zero_action_pathway(self)
        except Exception as e:  # noqa: BLE001 - never let the patch break model build
            print(f"[train.py] WARNING: action zero-init failed: {e}; leaving stock init")

    Cosmos3VFMNetwork.init_weights = _init_weights_zero_action
    print("[train.py] patched Cosmos3VFMNetwork.init_weights: action zero-init")


_patch_action_init()

# Stronger action conditioning (the final "film" recipe): two-tower encoder, CFG
# dropout, frame-aligned injection + per-frame FiLM, and condition-forcing. No-op
# unless the RYNNWORLD_ACTION_* / RYNNWORLD_COND_FORCE env vars are set. Must be registered
# before the runpy hand-off so the structure is present before FSDP.
from rynnworld_latent.action_conditioning import (  # noqa: E402
    register_action_conditioning_patches,
    register_cond_force_patches,
)

register_action_conditioning_patches()
register_cond_force_patches()


def _patch_dynamo_suppress_errors() -> None:
    """Make torch.compile failures non-fatal (fall back to eager).

    On some torch builds (e.g. 2.6) dynamo dies while tracing through the
    activation-checkpoint wrapper::

        torch._dynamo.exc.Unsupported: Observed exception
        from user code: checkpoint_wrapper.py ... return self.checkpoint_fn(...)

    which aborts every rank at the first training step. The recipe TOMLs already set
    ``[model.compile] enabled = false``, but this is a cheap safety net for any
    compiled region that slips through, and is exactly what the error message
    recommends. Set ``RYNNWORLD_DYNAMO_STRICT=1`` to keep such failures fatal.
    """
    if os.environ.get("RYNNWORLD_DYNAMO_STRICT") == "1":
        print("[train.py] RYNNWORLD_DYNAMO_STRICT=1; leaving dynamo errors fatal")
        return
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        print("[train.py] dynamo suppress_errors=True (compile failures fall back to eager)")
    except Exception as e:  # noqa: BLE001 - purely defensive
        print(f"[train.py] could not set dynamo suppress_errors: {e}")


_patch_dynamo_suppress_errors()


def _patch_wandb_stdout() -> None:
    """Mirror wandb-only callback metrics to the training log.

    MFU (per-GPU TFLOPS, including the VAE's share), the sequence-packing padding
    ratio and the norm monitor all compute their numbers on schedule but only hand
    them to ``wandb.log`` -- and this recipe runs with ``wandb_mode: disabled``,
    so they are computed and then discarded. The callbacks guard with
    ``if wandb.run is not None``, so give them a dummy run and replace
    ``wandb.log`` with a print. Rank 0 only. RYNNWORLD_WANDB_STDOUT=0 disables.

    ``wandb.init`` must be wrapped, not just ``wandb.log``: this runs at import
    time, but ``init_wandb`` (cosmos_framework/utils/wandb_util.py) calls
    ``wandb.init`` later, and init rebinds the module-level ``wandb.log`` to the
    bound method of the run it installs -- a truthy ``NoopRun`` when disabled.
    That silently undoes an import-time patch, and because the gate is a truthiness
    check the callbacks then keep computing and keep discarding. The dummy run only
    has to survive until init replaces it; ``wandb_util.py:118`` calls
    ``wandb.run.config.update``, which ``_DummyRun`` does not implement.
    """
    if os.environ.get("RYNNWORLD_WANDB_STDOUT", "1") != "1":
        return
    try:
        import wandb
    except Exception as e:  # noqa: BLE001 - purely defensive
        print(f"[train.py] wandb stdout patch skipped (no wandb): {e}")
        return

    class _DummyRun:
        """Truthy stand-in that absorbs any attribute access or call, to any depth.

        Truthiness is not the whole contract: cosmos's ``on_train_end`` callback calls
        ``wandb.finish()`` -> ``wandb.run.finish(exit_code=..., quiet=...)``, and
        ``wandb_util`` does ``wandb.run.config.update(...)`` -- two levels deep. A bare
        ``__slots__`` object raises ``AttributeError: '_DummyRun' object has no
        attribute 'finish'`` at teardown, which would abort an otherwise-successful run
        *after* its final checkpoint save. So absorb every attribute access and call.
        Live until ``wandb.init`` replaces it, and again at teardown on ranks where
        wandb restored or never installed a real run.
        """

        __slots__ = ()

        def __getattr__(self, name):
            return _DummyRun()

        def __call__(self, *args, **kwargs):
            return _DummyRun()

    def _log_to_stdout(data, *args, **kwargs):
        try:
            if int(os.environ.get("RANK", "0")) != 0:
                return
            parts = [f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in (data or {}).items()]
            print(f"[metrics] step={kwargs.get('step')} " + " | ".join(parts), flush=True)
        except Exception:  # noqa: BLE001 - never break training over a print
            pass

    def _install() -> None:
        if wandb.run is None:
            wandb.run = _DummyRun()
        wandb.log = _log_to_stdout

    _install()

    _real_init = wandb.init

    def _init_then_install(*args, **kwargs):
        run = _real_init(*args, **kwargs)
        _install()
        return run

    wandb.init = _init_then_install
    print("[train.py] wandb-only metrics (mfu/padding/norm) mirrored to stdout")


_patch_wandb_stdout()


def _patch_dcp_save_planner_compat() -> None:
    """Fix checkpoint saving on torch 2.6.

    cosmos's ``CustomSavePlanner.__init__`` (checkpoint/dcp.py) always forwards
    ``enable_plan_caching`` to ``torch...DefaultSavePlanner.__init__``, but torch
    2.6's DefaultSavePlanner has no such parameter (plan caching was added in a
    later torch). So the FIRST checkpoint save (iter == save_iter) raises
    ``TypeError: DefaultSavePlanner.__init__() got an unexpected keyword argument
    'enable_plan_caching'`` and kills every rank — which is why no checkpoint was
    ever written. Plan caching is only a save-time speedup, so we drop it on torch
    versions that lack it. No-op where torch already supports it (e.g. torch 2.10).
    """
    try:
        import inspect
        import torch.distributed.checkpoint.default_planner as _dp
        import cosmos_framework.checkpoint.dcp as _cdcp
    except Exception as e:  # noqa: BLE001
        print(f"[train.py] skip dcp save-planner patch: {e}")
        return

    if "enable_plan_caching" in inspect.signature(_dp.DefaultSavePlanner.__init__).parameters:
        return  # torch supports it; nothing to do

    _Base = _dp.DefaultSavePlanner

    def _compat_init(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_save_to_lowest_rank: bool = False,
        save_reg_to_ema: bool = False,
        enable_plan_caching: bool = False,  # accepted but ignored on this torch
        cache_plans_key=None,
    ) -> None:
        _Base.__init__(
            self,
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
        )
        if cache_plans_key is not None:
            self._cached_plans_key = cache_plans_key
        self.save_reg_to_ema = save_reg_to_ema

    _cdcp.CustomSavePlanner.__init__ = _compat_init
    print("[train.py] patched CustomSavePlanner.__init__ (dropped enable_plan_caching for torch 2.6)")


_patch_dcp_save_planner_compat()


def _ensure_te_free_optimizer() -> None:
    """Fall back to a fused AdamW optimizer when ``transformer_engine`` is absent.

    cosmos's default optimizer is ``FusedAdam``, which is provided by NVIDIA's
    ``transformer_engine`` / ``apex`` and only ships inside their containers. A
    stock ``pip install -r requirements.txt`` environment has neither, so the
    default config crashes at optimizer construction with
    ``ModuleNotFoundError: No module named 'transformer_engine'``. PyTorch's
    native fused AdamW (``optimizer_type=adamw`` + ``fused=true``) is numerically
    equivalent for our purposes and needs no extra dependency, so we inject it
    automatically. This mirrors the existing auto-patches around ``flash_attn``
    and ``lerobot``. A user who pins ``optimizer.optimizer_type=...`` themselves
    keeps their choice.
    """
    try:
        import transformer_engine  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    if any(a.startswith("optimizer.optimizer_type=") for a in sys.argv[1:]):
        return  # user pinned an optimizer explicitly; respect it

    sys.argv += ["optimizer.optimizer_type=adamw", "optimizer.fused=true"]
    print(
        "[train.py] transformer_engine unavailable -> optimizer fallback: "
        "optimizer.optimizer_type=adamw optimizer.fused=true"
    )


_ensure_te_free_optimizer()


# Execute the cosmos train script as __main__; sys.argv (--sft-toml, overrides)
# is passed through unchanged.
runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
