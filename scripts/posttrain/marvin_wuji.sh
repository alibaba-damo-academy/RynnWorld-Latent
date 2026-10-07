#!/bin/bash
# ============================================================================
# Marvin-WUJI — downstream embodiment post-training (quickstart).
#
# Fine-tunes the released RynnWorld-Latent film checkpoint on Marvin-WUJI robot
# teleop data via the `rynnworld_latent_edge_posttrain` recipe. It ships with ONE
# real bundled Marvin-WUJI sample (data/marvin_wuji/) so you can confirm the whole
# path runs end to end. For real adaptation, point MANIFEST_DIR at your own
# Marvin-WUJI corpus in the documented manifest format (README "Data format").
#
# Prereqs (see README "Downstream embodiment post-training"):
#   BASE_CHECKPOINT_PATH   a trained-film DCP dir — e.g. the output of
#                          scripts/checkpoints/convert_released_to_dcp.py run on
#                          the released weights (NOT the Cosmos3-Edge base).
#   WAN_VAE_PATH           Wan2.2_VAE.pth (gated nvidia/Cosmos3-Edge).
#   IMAGINAIRE_OUTPUT_ROOT optional; defaults to ./outputs (set by train.sh).
#
# Usage:
#   NGPU=8 bash scripts/posttrain/marvin_wuji.sh              # full run (3000 iter)
#   NGPU=1 MAX_ITER=3 bash scripts/posttrain/marvin_wuji.sh   # quick smoke
#
# On a stock requirements.txt env (no transformer_engine), scripts/train.py
# auto-injects a fused-AdamW optimizer, so no manual override is needed.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
EMBODIMENT="marvin_wuji"

# Data: the bundled single-sample manifest for this embodiment. Override to post-train
# on your own Marvin-WUJI corpus:  MANIFEST_DIR=/path/to/yours bash "$0"
export MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_ROOT/data/$EMBODIMENT/manifest}"

# Film switches the released checkpoint was trained with. These MUST match or the
# model structure mismatches; they are identical for every downstream embodiment
# because all of them warm-start from the same released film checkpoint.
export RYNNWORLD_TEXT_FREE=1 \
       RYNNWORLD_ACTION_CFG_DROPOUT=0.1 RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1 \
       RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1 RYNNWORLD_COND_FORCE=0.3 \
       RYNNWORLD_COND_FORCE_FLOOR=0.75 RYNNWORLD_ACTION_FRAME_INJECT=1 \
       RYNNWORLD_ACTION_TOWER_SPLIT=1 RYNNWORLD_ACTION_FILM=1

: "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH to a trained-film DCP dir (output of scripts/checkpoints/convert_released_to_dcp.py)}"
: "${WAN_VAE_PATH:?set WAN_VAE_PATH to Wan2.2_VAE.pth}"

# Optional short run. MAX_ITER caps iterations AND (required) the scheduler cycle,
# which must equal max_iter or the LR anneal never completes.
OVERRIDES=()
if [ -n "${MAX_ITER:-}" ]; then
    OVERRIDES+=(trainer.max_iter="$MAX_ITER" "scheduler.cycle_lengths=[$MAX_ITER]")
fi

echo "=== Marvin-WUJI post-train ==="
echo "  MANIFEST_DIR:          $MANIFEST_DIR"
echo "  BASE_CHECKPOINT_PATH:  $BASE_CHECKPOINT_PATH"
echo "  NGPU=${NGPU:-8}  MAX_ITER=${MAX_ITER:-<toml default 3000>}"

NGPU="${NGPU:-8}" TOML="${TOML:-configs/posttrain/downstream.toml}" \
    bash "$PROJECT_ROOT/scripts/train.sh" ${OVERRIDES[@]+"${OVERRIDES[@]}"}
