#!/bin/zsh
# Release gate step 2 (docs/release.md), run on the benchmark machine at 797f0ac.
# Steps run in order and stop at the first failure. Launched as:
#   caffeinate -is zsh runs/release-validation-2026-09-21/run.sh
cd "$(dirname "$0")/../.."
R=runs/release-validation-2026-09-21
source .venv/bin/activate
export K3_CKPT=/Volumes/USB4_2TBB/Kimi-K3
export K3_CACHE=$HOME/k3_dense
export K3_REVISION=c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721
export K3_MANIFEST="$PWD/manifest.jsonl.gz"

stamp() { echo "[$(date -u +%FT%TZ)] $*" >> $R/timeline.txt; sysctl vm.swapusage >> $R/timeline.txt; }
step() {
  local name=$1; shift
  if pgrep -fl 'air_llm.serve|run_long_job|bench' >> $R/timeline.txt; then
    stamp "ABORT before $name: another model/bench process is running"; exit 3
  fi
  stamp "start $name: $*"
  "$@" > $R/$name.log 2>&1
  local rc=$?
  stamp "end $name rc=$rc"
  if [[ $rc -ne 0 ]]; then stamp "STOP: $name failed"; exit $rc; fi
}

step 4a-run_all           python tests/run_all.py
step 4b-smoke             python -m air_llm.run --model "$K3_CKPT" --manifest "$K3_MANIFEST" \
                            --cache "$K3_CACHE" --mode runtime --resident-gb 8 --max-memory-gb 48 \
                            --prompt 'The capital of France is' --max-new-tokens 4 \
                            --check-finite --validation-log $R/smoke.jsonl
step 4c-numerical_modes   python tests/test_numerical_modes.py
# run_all.py --include-slow == 4a plus test_full_parity run with no arguments under this
# environment; run it directly so its full output (cosines, routers, token) is kept.
step 4d-full_parity       python tests/test_full_parity.py
stamp "ALL DONE"
