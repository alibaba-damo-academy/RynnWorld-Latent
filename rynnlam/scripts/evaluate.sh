#!/usr/bin/env bash
# Public single-checkpoint evaluation entrypoint; all data/output paths explicit.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/../evaluate.py" "$@"
