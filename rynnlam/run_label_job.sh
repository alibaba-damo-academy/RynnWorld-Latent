#!/bin/bash
# =============================================================================
# One latent-action labeling job = one A800 card. Submit this 250 times.
#
# The only thing that must differ between the 250 jobs is ID (0..SLICE-1): it
# selects which 1/SLICE slice of every dataset this job labels. If all 250 jobs
# ran with the same ID they would redo one slice and leave the other 249/250 of
# the corpus unlabeled, so ID resolution is strict and fails loudly.
#
# ID resolution order:
#   1. $ID, if you set it explicitly per job
#   2. a platform job-index env var (see JOB_INDEX_VARS below)
#   3. nothing -> exit 2 with a message, never silently fall back to 0
#
# Usage:
#   ID=$i bash run_label_job.sh                    # explicit index
#   bash run_label_job.sh                          # index from the platform env
# =============================================================================
set -uo pipefail

SLICE="${SLICE:-250}"
CKPT="${CKPT:-/path/to/lam_v2_checkpoint.pt}"
RYNNLAM="${RYNNLAM:-/path/to/RynnWorld-Latent/rynnlam}"
# The env that has PyAV + the rest of the rynnlam deps. The base interpreter does
# not, and a sweep run under it fails every view while still looking successful.
PYTHON="${PYTHON:-/opt/conda/envs/lam/bin/python}"

# Job-index env vars used by common batch platforms; the first one set wins.
JOB_INDEX_VARS="ID RANK JOB_INDEX TASK_INDEX TASK_ID INDEX POD_INDEX REPLICA_INDEX WORKER_ID"

resolved=""
source_var=""
for var in $JOB_INDEX_VARS; do
    value="${!var:-}"
    if [ -n "$value" ]; then
        resolved="$value"; source_var="$var"; break
    fi
done

if [ -z "$resolved" ]; then
    echo "[label-job] FATAL: no job index found. Set ID=<0..$((SLICE-1))> per job, or"
    echo "[label-job]        export one of: $JOB_INDEX_VARS"
    echo "[label-job]        Refusing to default to 0: 250 jobs sharing one ID would"
    echo "[label-job]        label 1/$SLICE of the corpus and skip the rest."
    exit 2
fi

case "$resolved" in
    ''|*[!0-9]*)
        echo "[label-job] FATAL: job index '$resolved' (from \$$source_var) is not an integer"
        exit 2 ;;
esac
if [ "$resolved" -lt 0 ] || [ "$resolved" -ge "$SLICE" ]; then
    echo "[label-job] FATAL: job index $resolved out of range 0..$((SLICE-1)) (SLICE=$SLICE)"
    exit 2
fi

if [ ! -x "$PYTHON" ]; then
    echo "[label-job] FATAL: interpreter not found or not executable: $PYTHON"
    echo "[label-job]        Set PYTHON=/path/to/python (needs PyAV: pip install -e '.[video]')"
    exit 2
fi

echo "[label-job] $(date '+%F %T') index=$resolved/$SLICE from \$$source_var  python=$PYTHON  ckpt=$CKPT"

exec env SLICE="$SLICE" ID="$resolved" CKPT="$CKPT" PYTHON="$PYTHON" \
    bash "$RYNNLAM/label_rynnvla_base.sh"
