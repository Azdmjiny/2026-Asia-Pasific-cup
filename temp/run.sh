#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run -n mcm python "$ROOT_DIR/temp/merge_and_audit.py"
