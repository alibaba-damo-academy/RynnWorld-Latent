#!/bin/bash
# Convert Cosmos3-Edge (HuggingFace/safetensors) -> DCP for cosmos SFT training.
#
# cosmos SFT uses ckpt_type=dcp and loads checkpoint.load_path via the
# DistributedCheckpointer, which expects <load_path>/model/*.distcp + .metadata.
# The shipped Cosmos3-Edge is HF/safetensors, so convert it first.
#
# We use scripts/checkpoints/convert_edge_to_dcp.py (NOT the stock convert_model_to_dcp),
# because the stock converter builds the model config from the checkpoint's
# partial config.json and crashes with "Missing key ema". Our script builds a
# complete config from EDGE_MODEL_CONFIG and loads the HF weights correctly.
#
# Writes to local /tmp first (FUSE/network FS may not support DCP's writes),
# then copies the result to the FUSE destination so training jobs can read it.
#
# Usage:
#   HF_SRC=/path/to/Cosmos3-Edge DCP_DST=/path/to/Cosmos3-Edge-dcp \
#       bash scripts/checkpoints/convert_edge_to_dcp.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COSMOS_ROOT="${COSMOS_ROOT:-$PROJECT_ROOT/third_party/cosmos-framework}"
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        echo "Usage: HF_SRC=/path/Cosmos3-Edge DCP_DST=/path/Cosmos3-Edge-dcp bash scripts/checkpoints/convert_edge_to_dcp.sh"
        echo "Optional environment: COSMOS_ROOT, WAN_VAE. Conversion runs on CPU."
        exit 0
    fi
done
: "${HF_SRC:?set HF_SRC to the Cosmos3-Edge HuggingFace dir}"
WAN_VAE="${WAN_VAE:-$COSMOS_ROOT/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth}"
: "${DCP_DST:?set DCP_DST to the output DCP checkpoint dir}"

export PYTHONPATH="$PROJECT_ROOT:$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_DEVICE=cpu

# Dependency guard: some cosmos imports pull in lerobot, which needs av<16.
if ! python -c "import av; _ = av.option" 2>/dev/null; then
    echo "[deps] av.option missing; installing av<16"
    pip install --quiet "av>=15.0.0,<16.0.0" || pip install --quiet "av<16.0.0" || true
fi

echo "=== Cosmos3-Edge -> DCP ==="
echo "  HF_SRC:  $HF_SRC"
echo "  WAN_VAE: $WAN_VAE"
echo "  DCP_DST: $DCP_DST"

# Convert into a local /tmp dir (FUSE-safe), then copy to the destination.
TMP_DST="$(mktemp -d /tmp/edge_dcp.XXXXXX)"
trap 'rm -rf "$TMP_DST"' EXIT

python "$PROJECT_ROOT/scripts/checkpoints/convert_edge_to_dcp.py" \
    --hf-src "$HF_SRC" \
    --vae "$WAN_VAE" \
    --out "$TMP_DST"

# Copy result to the FUSE destination (parent of model/).
mkdir -p "$DCP_DST"
cp -r "$TMP_DST/." "$DCP_DST/"

echo "=== Done. DCP checkpoint at: $DCP_DST ==="
ls -la "$DCP_DST"
ls -la "$DCP_DST/model" | head
