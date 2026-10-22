#!/usr/bin/env bash
set -euo pipefail
WORKFLOW=diagnose
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"
