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

cd /workspace
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
git fetch origin pull/1723/head:pr-1723
git checkout pr-1723
pip install -r requirements-hpu.txt
python setup.py develop
pip install datasets

export EXPERIMENTAL_WEIGHT_SHARING=0
export PT_HPU_LAZY_MODE=1
export VLLM_DECODE_BLOCK_BUCKET_MAX=2304
export VLLM_DECODE_BLOCK_BUCKET_STEP=256
export VLLM_DECODE_BS_BUCKET_STEP=32
export VLLM_DELAYED_SAMPLING=true
export VLLM_GRAPH_PROMPT_RATIO=0.2
export VLLM_GRAPH_RESERVED_MEM=0.04
export VLLM_PROMPT_BS_BUCKET_STEP=32
export VLLM_PROMPT_SEQ_BUCKET_MAX=4352
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_SKIP_WARMUP=false

set -x

python -m vllm.entrypoints.openai.api_server \
  --port $PORT \
  --model $MODEL \
  --block-size 128 \
  --dtype bfloat16 \
  --tensor-parallel-size=$TP \
  --max-model-len $MAX_MODEL_LEN  \
  --gpu-memory-util 0.95 \
  --use-padding-aware-scheduling \
  --max-num-seqs 192 \
  --max-num-prefill-seqs 16 \
  --num_scheduler_steps 1 \
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
