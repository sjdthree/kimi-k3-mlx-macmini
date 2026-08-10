#!/bin/zsh
# resident-gb sweep, with the process hygiene the first attempt lacked.
#
# What went wrong before: `pkill -f air_llm.serve` was ignored (the server sat in an
# uninterruptible read and never handled SIGTERM), so a stale server kept running while
# the next one started. Two servers then split the same USB4 disk, and the measurement
# recorded roughly half throughput as if it were a property of the config.
#
# So: never assume a kill worked. Escalate to SIGKILL, then ASSERT the count is zero
# before starting anything, and abort loudly rather than measure a contaminated machine.
set -u
PY=/opt/homebrew/Caskroom/miniforge/base/envs/airllm/bin/python3.12
REPO=~/kimi-k3-mlx
cd $REPO
TOKENS=${TOKENS:-40}

stop_all() {
  pkill -f "air_llm.serve" 2>/dev/null
  pkill -f "run_long_job"  2>/dev/null
  for i in {1..15}; do pgrep -f "air_llm.serve|run_long_job" >/dev/null || break; sleep 1; done
  # SIGTERM is advisory here; the server can be stuck in a read that never handles it
  pkill -9 -f "air_llm.serve" 2>/dev/null
  pkill -9 -f "run_long_job"  2>/dev/null
  for i in {1..15}; do pgrep -f "air_llm.serve|run_long_job" >/dev/null || break; sleep 1; done
  local n=$(pgrep -f "air_llm.serve|run_long_job" | wc -l | tr -d ' ')
  if [ "$n" != "0" ]; then
    echo "FATAL: $n process(es) survived SIGKILL — refusing to measure a dirty machine"
    pgrep -fl "air_llm.serve|run_long_job"
    exit 1
  fi
  echo "  clean: 0 servers, 0 jobs"
}

run_one() {
  local GB=$1
  local OUT=$REPO/runs/sweep-r${GB}
  echo "\n========== resident ${GB} GiB  ($(date +%H:%M:%S)) =========="
  stop_all
  rm -rf $OUT
  vm_stat | awk '/Pages free/{printf "  free RAM before: %.2f GB\n",$3*16384/1073741824}'

  export AIR_LLM_API_KEYS=sk-k3
  nohup $PY -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
      --manifest manifest_local.jsonl.gz --cache $HOME/k3_dense \
      --mode runtime --resident-gb $GB --max-memory-gb 58 --port 8080 \
      --max-output-tokens 3000 --token-log $OUT/tokens.jsonl \
      > /tmp/sweep-r${GB}.log 2>&1 &
  local SPID=$!

  local n=$(pgrep -f "air_llm.serve" | wc -l | tr -d ' ')
  [ "$n" = "1" ] || { echo "FATAL: expected exactly 1 server, found $n"; exit 1; }

  for i in {1..90}; do curl -sf -m 5 http://127.0.0.1:8080/readyz >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 5 http://127.0.0.1:8080/readyz >/dev/null || { echo "FATAL: server never ready"; exit 1; }

  $PY tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
      --out $OUT --max-tokens $TOKENS >/dev/null 2>&1

  # re-assert: exactly one server for the whole measurement
  n=$(pgrep -f "air_llm.serve" | wc -l | tr -d ' ')
  [ "$n" = "1" ] || echo "  WARNING: $n servers at end — result suspect"

  $PY - <<PYEOF
import json
d=[json.loads(l) for l in open("$OUT/tokens.jsonl")]
dec=[r for r in d if r['step_tokens']==1]
t=[r['elapsed_s'] for r in dec][1:]          # drop the warming token
if t:
    hr=dec[-1].get('cache_hit_rate',0); pk=dec[-1].get('peak_bytes',0)/2**30
    br=dec[-1]['bytes_read']/2**30
    print(f"  RESULT resident=$GB  n={len(t)}  mean={sum(t)/len(t):.1f}s  "
          f"min={min(t):.1f} max={max(t):.1f}")
    print(f"         hit_rate={hr:.3f}  peak={pk:.1f}GiB  "
          f"throughput={br/(sum(t)/len(t)):.2f} GiB/s")
PYEOF
}

for gb in ${=CONFIGS:-24 32 40}; do run_one $gb; done
echo "\n========== sweep done $(date +%H:%M:%S) =========="
stop_all
