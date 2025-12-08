#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# PORT
# TP
# CONC
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# NUM_PROMPTS

export VLLM_USE_AITER_UNIFIED_ATTENTION=1

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

cat > vll << EOF
#!/usr/bin/python
import sys
from vllm.entrypoints.cli.main import main
if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
EOF
chmod 755 vll

ls -lrt
cat vll



set -x
./vll serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.85 \
--max-model-len $MAX_MODEL_LEN \
--max-seq-len-to-capture $MAX_MODEL_LEN \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /gitworkspace/
