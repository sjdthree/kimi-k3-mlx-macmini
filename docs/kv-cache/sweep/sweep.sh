#!/bin/sh
# Portable replacement; arguments are passed explicitly. No output is deleted.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
exec "${PYTHON:-python3}" "$ROOT/tools/benchmark_run.py" "$@"
