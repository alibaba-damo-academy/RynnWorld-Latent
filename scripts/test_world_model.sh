#!/usr/bin/env bash
# =============================================================================
# RynnWorld-Latent — end-to-end train / test driver.
#
# Runs the release's real code paths against the bundled data/ sample, in the
# order you would want them checked:
#
#   data         CPU only. Load data/manifest, pull every sample through the
#                dataset, assert shapes / dtypes / normalization range. No
#                weights, no GPU. Start here.
#   lam          RynnLAM inference: re-encode data/videos and compare against the
#                shipped data/latents (scripts/test_lam_inference.py). Needs a
#                LAM checkpoint + GPU.
#   train-smoke  3 training iterations on the bundled sample, 1 GPU. Proves the
#                whole chain: DCP base load -> patches -> dataset -> forward ->
#                backward -> optimizer step.
#   train        50 iterations, to confirm the loss actually moves.
#   infer        Roll out a trained checkpoint on the bundled sample and report
#                PSNR vs GT, the static-first-frame baseline, and the motion
#                ratio (scripts/inference/rollout.py).
#   all          data, train-smoke, train, infer (lam needs --lam-ckpt).
#
# Usage:
#   bash scripts/test_world_model.sh --stage data
#   bash scripts/test_world_model.sh --stage train-smoke
#   bash scripts/test_world_model.sh --stage infer --checkpoint /path/to/iter_000005000
#   bash scripts/test_world_model.sh --stage all
#
# Required env for the stages that load a model:
#   BASE_CHECKPOINT_PATH   Cosmos3-Edge DCP dir   (scripts/setup/download_weights.sh)
#   WAN_VAE_PATH           Wan2.2_VAE.pth         (printed by the same script)
# Optional:
#   RYNNLAM_CKPT           LAM checkpoint, for --stage lam
#   OUT_ROOT               where to write (default $TMPDIR/rynnworld_latent_checks)
#   QUICKSTART_NGPU        GPUs for the training stages (default 1)
#   NUM_STEPS / GUIDANCE   diffusion steps (25) / action guidance (1.5) for infer
#   --no-film              skip the film action-conditioning switches
#
# The film switches are exported by default because that is the released recipe;
# a checkpoint trained with them MUST be evaluated with them, or frame_gate /
# frame_film are missing from the model structure and the load reports gaps.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

STAGE="data"
CHECKPOINT="${CHECKPOINT:-}"
FILM=1
DATA_DIR="$REPO/data"

while [ $# -gt 0 ]; do
    case "$1" in
        --stage)      STAGE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --no-film)    FILM=0; shift ;;
        -h|--help)    sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Local disk by default: the training stages end with a full checkpoint write
# (~20 GB for Edge) and the inference stage writes mp4s. Point OUT_ROOT at the repo
# only if you actually want those artifacts kept there.
OUT_ROOT="${OUT_ROOT:-${TMPDIR:-/tmp}/rynnworld_latent_checks}"
MANIFEST_DIR="${MANIFEST_DIR:-$DATA_DIR/manifest}"
mkdir -p "$OUT_ROOT"
export PYTHONPATH="$REPO:$REPO/third_party/cosmos-framework:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

if [ ! -f "$MANIFEST_DIR/summary.json" ]; then
    echo "ERROR: no manifest at $MANIFEST_DIR (expected the bundled data/ sample)" >&2
    exit 2
fi

if [ "$FILM" = 1 ]; then
    export RYNNWORLD_TEXT_FREE=1
    export RYNNWORLD_ACTION_CFG_DROPOUT=0.1
    export RYNNWORLD_ACTION_CFG_DROPOUT_HAND=0.1
    export RYNNWORLD_ACTION_CFG_DROPOUT_CAM=0.1
    export RYNNWORLD_COND_FORCE=0.3
    export RYNNWORLD_COND_FORCE_FLOOR=0.75
    export RYNNWORLD_ACTION_FRAME_INJECT=1
    export RYNNWORLD_ACTION_TOWER_SPLIT=1
    export RYNNWORLD_ACTION_FILM=1
fi

hr() { printf '\n%s\n' "=============== $* ==============="; }

need_weights() {
    : "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH (scripts/setup/download_weights.sh prints it)}"
    : "${WAN_VAE_PATH:?set WAN_VAE_PATH to Wan2.2_VAE.pth}"
    export BASE_CHECKPOINT_PATH WAN_VAE_PATH
}

stage_data() {
    hr "STAGE data — CPU-only dataset check on $MANIFEST_DIR"
    python3 - "$MANIFEST_DIR" <<'PY'
import sys
from rynnworld_latent.dataset import ACTION_DIM, DOMAIN_ID
from rynnworld_latent.manifest_dataset import RynnWorldManifestDataset

mdir = sys.argv[1]
ds = RynnWorldManifestDataset(manifest_dir=mdir, staged_root=None, fps=10.0,
                              action_normalization="quantile")
print(f"  records        : {len(ds)}")
print(f"  action_dim     : {ds.action_dim} (expected {ACTION_DIM})")
assert ds.action_dim == ACTION_DIM, ds.action_dim

for i in range(len(ds)):
    s = ds._load_one(i)          # no retry fallback: exercise THIS record
    v, a = s["video"], s["action"]
    # record tuple slots: (vid, src, lat, start, n_frames, n_actions, ...)
    nf, na = ds._recs[i][4], ds._recs[i][5]
    assert v.ndim == 4 and v.shape[0] == 3 and v.shape[1] == nf, v.shape
    assert v.dtype.is_floating_point is False and v.min() >= 0 and v.max() <= 255
    assert tuple(a.shape) == (na, ACTION_DIM), a.shape
    assert a.min() >= -1.0 and a.max() <= 1.0, (float(a.min()), float(a.max()))
    assert int(s["domain_id"]) == DOMAIN_ID
    assert s["mode"] == "forward_dynamics"
    print(f"  [{i}] video={tuple(v.shape)} action={tuple(a.shape)} "
          f"cap={s['ai_caption'][:38]!r} fps={int(s['conditioning_fps'])}")
print("  DATA STAGE PASSED")
PY
}

stage_lam() {
    hr "STAGE lam — RynnLAM inference vs the bundled latents"
    : "${RYNNLAM_CKPT:?set RYNNLAM_CKPT to a RynnLAM checkpoint (or use --stage without lam)}"
    python3 scripts/test_lam_inference.py --lam-ckpt "$RYNNLAM_CKPT" --data-dir "$DATA_DIR"
}

stage_train() {  # $1 = iterations
    local iters="$1"
    local ngpu="${QUICKSTART_NGPU:-1}"
    hr "STAGE train — $iters iterations on the bundled sample (NGPU=$ngpu)"
    need_weights
    export STAGED_ROOT="${STAGED_ROOT:-}"
    export IMAGINAIRE_OUTPUT_ROOT="$OUT_ROOT/train$iters"
    export QUICKSTART_ITERS="$iters" QUICKSTART_NGPU="$ngpu"
    local log="$OUT_ROOT/train_$iters.log"
    if [ "$MANIFEST_DIR" = "$DATA_DIR/manifest" ]; then
        # train_8gpu.sh's quickstart triggers on MANIFEST_DIR being *unset* and then
        # points at the bundled data/ itself. Exporting it here would silently
        # select the 8-GPU full-parameter recipe instead, so hide it.
        env -u MANIFEST_DIR bash scripts/train_8gpu.sh 2>&1 | tee "$log"
    else
        MANIFEST_DIR="$MANIFEST_DIR" NGPU="$ngpu" \
        TOML="${TOML:-configs/examples/edge_manifest_local.toml}" \
            bash scripts/train_8gpu.sh \
            optimizer.optimizer_type=adamw optimizer.fused=true \
            trainer.max_iter="$iters" trainer.logging_iter=1 \
            "scheduler.cycle_lengths=[$iters]" 2>&1 | tee "$log"
    fi
    echo "  log: $log"
    grep -E 'loss|patches ON|action init|Error|Traceback' "$log" | tail -20 || true
}

stage_infer() {
    hr "STAGE infer — rollout on the bundled sample"
    need_weights
    : "${CHECKPOINT:?set --checkpoint <iter_XXXXXXXX dir> (or \$CHECKPOINT)}"
    local out="$OUT_ROOT/rollout"
    python3 scripts/inference/rollout.py \
        --manifest-dir "$MANIFEST_DIR" \
        --staged-root "${STAGED_ROOT:-}" \
        --checkpoint "$CHECKPOINT" \
        --indices 0,1,2 \
        --num-steps "${NUM_STEPS:-25}" \
        --guidance "${GUIDANCE:-1.5}" \
        --out "$out" 2>&1 | tee "$OUT_ROOT/infer.log"
    echo "  videos + log under: $out , $OUT_ROOT/infer.log"
}

case "$STAGE" in
    data)        stage_data ;;
    lam)         stage_lam ;;
    train-smoke) stage_train 3 ;;
    train)       stage_train 50 ;;
    infer)       stage_infer ;;
    all)         stage_data; stage_train 3; stage_train 50; stage_infer ;;
    *) echo "unknown --stage '$STAGE' (data|lam|train-smoke|train|infer|all)" >&2; exit 2 ;;
esac

hr "stage '$STAGE' finished"
