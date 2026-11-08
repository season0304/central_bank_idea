#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m wcb_4090.pipeline --triplet-only "$@"
