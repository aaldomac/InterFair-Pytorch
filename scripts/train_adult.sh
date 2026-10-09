#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config="${1:-configs/adult.yaml}"
if (( $# )); then shift; fi
exec "${PYTHON_BIN:-python}" -m scripts.run_experiment --config "$config" "$@"
