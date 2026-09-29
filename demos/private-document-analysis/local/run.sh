#!/bin/sh
# Run the three cases on the local implementation, one after another, R&D first.
# Expect days per case. Rerunning this script resumes wherever it stopped: finished
# cases return at once, an unfinished one continues from its last saved state.
#
#   export K3_CKPT=/path/to/Kimi-K3 K3_MANIFEST=/path/to/manifest.jsonl.gz
#   export K3_CACHE=/path/to/internal-ssd/k3_dense    # optional, see INSTALL.md
#   sh demos/private-document-analysis/local/run.sh [case ...]
#
# Settings: Moonshot's published sampling for single-step tasks (temperature 1.0,
# top-p 0.95), its default thinking effort (max), and the cloud run's 20,000-token cap.
set -eu
: "${K3_CKPT:?set K3_CKPT}" "${K3_MANIFEST:?set K3_MANIFEST}"
cd "$(dirname "$0")/../../.."
OUT=${OUT:-runs/demo-local}
CASES=${*:-rd_tradesecret semiconductor medical_ai}

for c in $CASES; do
  echo "=== $c -> $OUT/$c"
  # -i: no idle sleep while the job runs; -s: nor system sleep on AC power.
  caffeinate -is python tools/run_chat_job.py \
    --model "$K3_CKPT" --manifest "$K3_MANIFEST" ${K3_CACHE:+--cache "$K3_CACHE"} \
    --mode runtime --resident-gb 8 --max-memory-gb 48 \
    --messages "demos/private-document-analysis/local/inputs/$c.json" \
    --out "$OUT/$c" --max-tokens 20000 \
    --temperature 1.0 --top-p 0.95 --seed 0 --thinking-effort max
done
