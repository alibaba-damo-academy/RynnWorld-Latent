#!/usr/bin/env bash
# =============================================================================
# RynnWorld-Latent — single-node training submit script.
#
# Two ways to run it:
#
#   1) QUICKSTART (zero setup, proves the pipeline runs end to end):
#         bash scripts/setup/download_weights.sh           # gated weights, once
#         export BASE_CHECKPOINT_PATH=... WAN_VAE_PATH=... # printed by the above
#         bash scripts/train_8gpu.sh
#      With no MANIFEST_DIR set, this trains on the bundled data/ -- 3 REAL
#      RynnWorld-Latent v2 samples (81 frames / 20 latent actions each, shipped in-repo
#      at the production 480p geometry; provenance in the README). Only a few
#      samples -> 8 GPUs would starve, so the quickstart caps NGPU=1; override
#      with QUICKSTART_NGPU / QUICKSTART_ITERS.
#
#   2) Training on your own manifest, 8 GPUs:
#         export MANIFEST_DIR=/path/to/manifest STAGED_ROOT=/path/to/staged480
#         export BASE_CHECKPOINT_PATH=... WAN_VAE_PATH=...
#         NGPU=8 TOML=configs/train/edge_fullft.toml bash scripts/train_8gpu.sh
#
# Everything is single-node. Extra args are forwarded to scripts/train.py as
# Hydra overrides.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

NGPU="${NGPU:-8}"
TOML="${TOML:-configs/train/edge_fullft.toml}"
EXTRA=()

if [ -z "${MANIFEST_DIR:-}" ]; then
    echo "[train_8gpu] no MANIFEST_DIR -> QUICKSTART on bundled data/"
    if [ ! -f data/manifest/summary.json ]; then
        echo "[train_8gpu] ERROR: bundled data/manifest is missing; cannot quickstart." >&2
        exit 2
    fi
    export MANIFEST_DIR="$REPO/data/manifest"
    export STAGED_ROOT="${STAGED_ROOT:-}"
    TOML="${QUICKSTART_TOML:-configs/examples/edge_manifest_local.toml}"
    NGPU="${QUICKSTART_NGPU:-1}"
    ITERS="${QUICKSTART_ITERS:-3}"
    # transformer_engine's FusedAdam is usually absent off-cluster; use plain AdamW.
    # cycle_lengths must equal max_iter or the LR schedule never anneals, and the
    # recipe TOML pins a long cycle — so override both, not just max_iter.
    EXTRA=(optimizer.optimizer_type=adamw optimizer.fused=true
           trainer.max_iter="$ITERS" trainer.logging_iter=1
           "scheduler.cycle_lengths=[$ITERS]")
    # cosmos ALWAYS writes one full checkpoint when training ends (trainer/__init__.py
    # saves whenever `iteration % save_iter != 0`), so even a 3-iteration smoke leaves
    # a ~20 GB artifact for Edge — save_iter=999999 does not suppress it. Default the
    # output root to LOCAL disk: on a network mount that write takes many minutes
    # *after* "Done with training." is printed, which is indistinguishable from a hang.
    export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${TMPDIR:-/tmp}/rynnworld_latent_quickstart}"
    echo "[train_8gpu] quickstart output (incl. the end-of-run checkpoint, ~20 GB for Edge):"
    echo "[train_8gpu]   $IMAGINAIRE_OUTPUT_ROOT"
    echo "[train_8gpu] quickstart: NGPU=$NGPU TOML=$TOML iters=$ITERS"
fi

: "${BASE_CHECKPOINT_PATH:?download weights first: bash scripts/setup/download_weights.sh (then export BASE_CHECKPOINT_PATH)}"
: "${WAN_VAE_PATH:?set WAN_VAE_PATH to Wan2.2_VAE.pth (download_weights.sh prints it)}"

export NGPU
export MASTER_PORT="${MASTER_PORT:-29500}"
export TOML
echo "[train_8gpu] single-node submit: NGPU=$NGPU TOML=$TOML"
echo "[train_8gpu]   MANIFEST_DIR=${MANIFEST_DIR:-<unset>}"
echo "[train_8gpu]   BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
echo "[train_8gpu]   WAN_VAE_PATH=$WAN_VAE_PATH"
echo "[train_8gpu]   IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$REPO/outputs}"

exec bash scripts/train.sh "${EXTRA[@]}" "$@"
