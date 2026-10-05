#!/bin/bash
# RynnWorld-Latent training launch script (single-node torchrun).
#
# Env required:
#   MANIFEST_DIR (+ optional STAGED_ROOT), BASE_CHECKPOINT_PATH, WAN_VAE_PATH;
#   IMAGINAIRE_OUTPUT_ROOT defaults to outputs/.
#   COSMOS_ROOT defaults to the vendored third_party/cosmos-framework.
#
# For the "film" action-conditioning recipe, also export (see README):
#   RYNNWORLD_TEXT_FREE=1 \
#   RYNNWORLD_ACTION_CFG_DROPOUT=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1 \
#   RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1 RYNNWORLD_COND_FORCE=0.3 \
#   RYNNWORLD_COND_FORCE_FLOOR=0.75 RYNNWORLD_ACTION_FRAME_INJECT=1 \
#   RYNNWORLD_ACTION_TOWER_SPLIT=1 RYNNWORLD_ACTION_FILM=1
#
# Usage:
#   bash scripts/train.sh              # 8-GPU single node
#   NGPU=4 bash scripts/train.sh       # 4-GPU
#   TOML=configs/examples/edge_manifest_local.toml NGPU=1 bash scripts/train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COSMOS_ROOT="${COSMOS_ROOT:-$PROJECT_ROOT/third_party/cosmos-framework}"

DEFAULT_TOML="configs/train/edge_fullft.toml"

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        echo "Usage: [NGPU=8] [TOML=$DEFAULT_TOML] bash scripts/train.sh [trainer arguments]"
        exec python "$PROJECT_ROOT/scripts/train.py" --help
    fi
done

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
# Recipe TOML (relative to PROJECT_ROOT). Override for a local/single-GPU config.
TOML="${TOML:-$DEFAULT_TOML}"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$COSMOS_ROOT:${PYTHONPATH:-}"

if [ -z "${MANIFEST_DIR:-}" ]; then
    echo "ERROR: set MANIFEST_DIR to a chunk manifest directory (e.g. the bundled data/manifest)." >&2
    exit 2
fi

echo "=== RynnWorld-Latent Training (Cosmos3-Edge, single node) ==="
echo "  MANIFEST_DIR:           $MANIFEST_DIR"
echo "  STAGED_ROOT:            ${STAGED_ROOT:-<empty: read original source videos>}"
echo "  BASE_CHECKPOINT_PATH:   ${BASE_CHECKPOINT_PATH:?}"
echo "  WAN_VAE_PATH:           ${WAN_VAE_PATH:?}"
echo "  IMAGINAIRE_OUTPUT_ROOT: $IMAGINAIRE_OUTPUT_ROOT"
echo "  TOML:                   $TOML"
echo "  NGPU=$NGPU"
echo "================================="

# Route through scripts/train.py (NOT `-m cosmos_framework.scripts.train`):
# train.py registers the rynnworld_latent_edge_manifest* experiments in Hydra's
# ConfigStore and applies the compat + action zero-init patches before delegating
# to the stock cosmos train script.
cd "$PROJECT_ROOT"
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NGPU" \
    scripts/train.py \
    --sft-toml "$PROJECT_ROOT/$TOML" \
    "$@"
