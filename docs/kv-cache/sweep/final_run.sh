#!/bin/zsh
# Final validation: the ORIGINAL baseline prompt and token target, on the branch, at the
# resident-gb the sweep chose. Answers the three questions directly:
#   a) same substantive result?   b) much faster?   c) no memory stress?
#
# Same process hygiene as the sweep -- a stale server competing for the USB4 disk is what
# made an earlier measurement unusable, and this run is ~20 hours.
set -u
GB=${1:?usage: final_run.sh <resident-gb>}
PY=/opt/homebrew/Caskroom/miniforge/base/envs/airllm/bin/python3.12
REPO=~/kimi-k3-mlx
OUT=$REPO/runs/final-validation
cd $REPO

echo "=== stopping everything ($(date +%H:%M:%S)) ==="
pkill -f "air_llm.serve" 2>/dev/null; pkill -f "run_long_job" 2>/dev/null
for i in {1..15}; do pgrep -f "air_llm.serve|run_long_job" >/dev/null || break; sleep 1; done
pkill -9 -f "air_llm.serve" 2>/dev/null; pkill -9 -f "run_long_job" 2>/dev/null
for i in {1..15}; do pgrep -f "air_llm.serve|run_long_job" >/dev/null || break; sleep 1; done
n=$(pgrep -f "air_llm.serve|run_long_job" | wc -l | tr -d ' ')
[ "$n" = "0" ] || { echo "FATAL: $n survived SIGKILL"; pgrep -fl "air_llm.serve|run_long_job"; exit 1; }
echo "  clean"
vm_stat | awk '/Pages free/{printf "  free RAM %.2f GB\n",$3*16384/1073741824}'
sysctl -n vm.swapusage | sed 's/^/  swap /'

rm -rf $OUT
export AIR_LLM_API_KEYS=sk-k3
echo "=== server: resident ${GB} GiB, ceiling 58 ==="
nohup $PY -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
    --manifest manifest_local.jsonl.gz --cache $HOME/k3_dense \
    --mode runtime --resident-gb $GB --max-memory-gb 58 --port 8080 \
    --max-output-tokens 3000 --token-log $OUT/tokens.jsonl \
    > /tmp/serve-final.log 2>&1 &
echo "  pid $!"
n=$(pgrep -f "air_llm.serve" | wc -l | tr -d ' ')
[ "$n" = "1" ] || { echo "FATAL: expected 1 server, found $n"; exit 1; }
for i in {1..90}; do curl -sf -m 5 http://127.0.0.1:8080/readyz >/dev/null 2>&1 && break; sleep 5; done
curl -sf -m 5 http://127.0.0.1:8080/readyz >/dev/null || { echo "FATAL: never ready"; exit 1; }
echo "  ready"

echo "=== launching: prompt 04, 2500 tokens (baseline: 2,197 generated in 51.0 h) ==="
nohup caffeinate -is $PY tools/run_long_job.py \
    --prompt-file prompts/04_polymath_multi_expert.md \
    --out $OUT --max-tokens 2500 > /tmp/final-job.log 2>&1 &
echo "  job pid $!"
sleep 20
cat $OUT/progress.json 2>/dev/null | $PY -c "import json,sys;p=json.load(sys.stdin);print('  state',p['state'])" 2>/dev/null
echo "=== launched $(date +%H:%M:%S); ~20 h at 32.8 s/token ==="
