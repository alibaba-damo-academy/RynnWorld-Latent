#!/usr/bin/env bash
# =============================================================================
# Download the pretrained weights RynnWorld-Latent fine-tunes from, and convert
# the Cosmos3-Edge backbone into the DCP layout cosmos-framework trains on.
#
# This repo ships CODE ONLY -- no model weights (see THIRD_PARTY_LICENSES.md).
# The weights are NVIDIA Cosmos3 checkpoints on the Hugging Face Hub:
#
#   nvidia/Cosmos3-Edge   3.4B backbone (Nemotron-2B MoT)
#   Wan2.2_VAE.pth        video tokenizer, bundled in the Cosmos3 release
#
# These are GATED repos: request access on the HF page and log in first
#   huggingface-cli login        # or: export HF_TOKEN=hf_...
#
# Usage:
#   bash scripts/setup/download_weights.sh
#   DEST=/data/cosmos bash scripts/setup/download_weights.sh
#
# Result:
#   $DEST/Cosmos3-Edge/        HF snapshot (safetensors + Wan2.2_VAE.pth)
#   $DEST/Cosmos3-Edge-dcp/    DCP checkpoint -> export BASE_CHECKPOINT_PATH=this
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="${DEST:-$REPO_ROOT/weights}"
mkdir -p "$DEST"

python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "ERROR: huggingface_hub not installed. pip install -U huggingface_hub" >&2; exit 1; }

snapshot() {  # snapshot <repo_id> -> prints local dir
    python3 - "$1" "$DEST" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
print(snapshot_download(repo_id=repo, local_dir=os.path.join(dest, repo.split("/")[-1]),
                        token=os.environ.get("HF_TOKEN")), end="")
PY
}

find_vae() {  # find_vae <dir> -> prints path to Wan2.2_VAE.pth
    find "$1" -name 'Wan2.2_VAE.pth' -type f 2>/dev/null | head -1
}

echo "[weights] downloading nvidia/Cosmos3-Edge ..."
hf_dir="$(snapshot nvidia/Cosmos3-Edge)"
vae="$(find_vae "$hf_dir")"
[ -n "$vae" ] || { echo "ERROR: Wan2.2_VAE.pth not found under $hf_dir" >&2; exit 1; }

out="$DEST/Cosmos3-Edge-dcp"
echo "[weights] converting Cosmos3-Edge -> DCP at $out"
COSMOS_DEVICE=cpu python3 "$REPO_ROOT/scripts/checkpoints/convert_edge_to_dcp.py" \
    --hf-src "$hf_dir" --vae "$vae" --out "$out"

echo "[weights] done. Export:"
echo "  export BASE_CHECKPOINT_PATH=$out"
echo "  export WAN_VAE_PATH=$vae"
