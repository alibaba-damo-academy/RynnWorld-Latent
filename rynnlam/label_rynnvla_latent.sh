#!/bin/bash
# =============================================================================
# Label ALL RynnVLA-Latent datasets with stride-4 latent action.
#
# Latent action = featsrc k_token (K=8 x 64 = 512d) from the b512 checkpoint.
# Stride-4: pair (t, t+4), start frames advance by 4 -> (0,4),(4,8),(8,12)...
# so 1 latent action per 4 RGB frames, matching the world model's Wan-VAE 4x
# temporal cadence (4 RGB frames -> 1 video latent -> 1 action).
#
# Serves: RynnWorld-Latent (the world model in this repository).
#
# 200-card parallelism: launch ONE process per card, each with its own ID.
#   SLICE = total number of processes (= cards), ID = this process index.
# The pipeline partitions episodes by --shard/--num-shards, so each process
# labels a disjoint 1/SLICE subset of every dataset's episodes.
#
# Resume-safe: atomic publish + protocol fingerprint. Re-running skips already
# labeled episodes; a stride/checkpoint/protocol mismatch is refused, not
# silently overwritten. SIGKILL never leaves a half-written label.
#
# Usage (per process, e.g. card i of 200):
#   SLICE=200 ID=$i CKPT=/path/to/b512_slim.pt bash label_rynnvla_latent.sh
# =============================================================================
set -uo pipefail

SLICE="${SLICE:-250}"                 # total parallel processes (= cards)
ID="${ID:-0}"                         # this process index, 0..SLICE-1
CKPT="${CKPT:?set CKPT to the validated + slimmed b512 checkpoint (.pt with config+model_state_dict)}"
DATA="${DATA:-/path/to/RynnVLA-Latent}"
OUT="${OUT:-/path/to/RynnVLA-Latent-ktoken608}"
RYNNLAM="${RYNNLAM:-/path/to/RynnWorld-Latent/rynnlam}"
PYTHON="${PYTHON:-python}"                 # same convention as scripts/evaluate.sh
SCRATCH="${SCRATCH:-/tmp/rynnlam-label-scratch-$ID}"   # per-process local scratch
REPRESENTATION="${REPRESENTATION:-ktoken_zcam}"        # ktoken_zcam=608d (k_token 512 + z 64 + camera 32)
GAP="${GAP:-4}"                                        # RGB frame interval within a pair
PAIR_STRIDE="${PAIR_STRIDE:-4}"                        # start-frame stride (1 latent / 4 frames)
BATCH_SIZE="${BATCH_SIZE:-32}"
PREFETCH="${PREFETCH:-2}"
CURSOR_EVERY="${CURSOR_EVERY:-20}"       # episodes between resume-cursor flushes;
                                         # a kill loses at most this many episodes

# RoVid-X clips are addressed as 'rovidx_tar://<md5>.mp4' inside tar shards, so
# resolving them needs the tar root plus the byte-offset index.
export ROVIDX_TAR_ROOT="${ROVIDX_TAR_ROOT:-/path/to/RoVid-X}"
export ROVIDX_INDEX_ROOT="${ROVIDX_INDEX_ROOT:-/path/to/RoVid-X_index}"

if [ ! -f "$CKPT" ]; then echo "[label] FATAL: checkpoint not found: $CKPT"; exit 2; fi
if [ ! -d "$DATA" ]; then echo "[label] FATAL: data dir not found: $DATA"; exit 2; fi
mkdir -p "$SCRATCH" "$OUT"
cd "$RYNNLAM" || { echo "[label] FATAL: cannot cd $RYNNLAM"; exit 2; }

# Every view is decoded through PyAV (rynnlam/video.py). Without it each view
# fails with "No module named 'av'", and because per-view failures do not abort
# the sweep the job would walk all datasets, record nothing but errors, and look
# like a normal run. Check once here rather than once per episode.
if ! "$PYTHON" -c "import av" >/dev/null 2>&1; then
    echo "[label] FATAL: $PYTHON cannot import 'av' (PyAV); no video would decode."
    echo "[label]        Fix: $PYTHON -m pip install -e '.[video]',"
    echo "[label]        or point PYTHON at an interpreter that has it."
    exit 2
fi

echo "[label $(date '+%F %T')] start shard $ID/$SLICE  python=$PYTHON  ckpt=$CKPT  repr=$REPRESENTATION  gap=$GAP stride=$PAIR_STRIDE  out=$OUT"

shopt -s nullglob
labeled=0; skipped=0; failed=0
# Stagger the dataset order by job ID. Every job walks the same directory, so
# without this all 250 jobs parse the same multi-gigabyte JSON (RoVid-X 3.6 GB,
# HowTo100M 4.4 GB, ~11-13 GB RSS each) at the same moment, spiking both OSS
# reads and node memory at once.
all_json=("$DATA"/*.json)
n_json=${#all_json[@]}
if [ "$n_json" -gt 0 ]; then
    offset=$(( ID % n_json ))
    jsons=("${all_json[@]:offset}" "${all_json[@]:0:offset}")
else
    jsons=()
fi
for ds_json in "${jsons[@]}"; do
    ds="$(basename "$ds_json" .json)"
    # Skip non-dataset artifacts (caption outputs, backups, builder scripts, remaps).
    case "$ds" in
        caption*|nocaption*|*_result|remapped|original_backup|build_*|rebuild_*|remap_*|*.bak)
            skipped=$((skipped+1)); continue ;;
    esac
    # Restart fast path: a finished dataset is skipped without loading its JSON
    # (the big ones are 1-4 GB and cost ~11 GB RSS to parse). Delete the marker to
    # re-run a dataset.
    done_marker="$OUT/done/$(printf '%s.shard%04d.json' "$ds" "$ID")"
    if [ -s "$done_marker" ]; then
        echo "[label] skip $ds (already done: $done_marker)"; skipped=$((skipped+1)); continue
    fi
    # A dataset JSON must be a list of {dataset,episode_id,views}; check only the
    # header so validating a 4 GB file does not parse it.
    if ! head -c 4096 "$ds_json" | grep -q '"views"'; then
        echo "[label] skip $ds (not an episode-list JSON)"; skipped=$((skipped+1)); continue
    fi
    echo "[label $(date '+%T')] === $ds  shard $ID/$SLICE ==="
    if "$PYTHON" label_latent_dense.py \
        --metadata "$ds_json" \
        --checkpoint "$CKPT" \
        --output-dir "$OUT" \
        --representation "$REPRESENTATION" \
        --gap "$GAP" --pair-stride "$PAIR_STRIDE" \
        --batch-size "$BATCH_SIZE" --prefetch "$PREFETCH" \
        --cursor-every "$CURSOR_EVERY" \
        --shard "$ID" --num-shards "$SLICE" \
        --scratch "$SCRATCH" \
        --device cuda --precision bf16; then
        labeled=$((labeled+1))
    else
        # Per-view failures are recorded in failed_shardNNNN.json and exit nonzero;
        # do not abort the whole sweep — continue to the next dataset.
        echo "[label] WARN: $ds shard $ID exited nonzero (see $OUT/failed_shard*.json); continuing"
        failed=$((failed+1))
    fi
done

echo "[label $(date '+%F %T')] DONE shard $ID/$SLICE  datasets_labeled=$labeled skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ]
