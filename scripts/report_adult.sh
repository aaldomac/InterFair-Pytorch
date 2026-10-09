#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
folder="${1:-experiments/adult}"
if (( $# )); then shift; fi
exec "${PYTHON_BIN:-python}" -m scripts.report_real "$folder" "$@"
