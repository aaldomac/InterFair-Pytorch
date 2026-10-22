#!/usr/bin/env bash
set -euo pipefail
WORKFLOW=audit
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"