#!/usr/bin/env bash
# Compare the Ollama sampling variants A-D (models/Modelfile.*) on the eval set.
# Resumable: runs whose results file already exists are skipped, so it can be
# stopped (Ctrl-C) and started again later. A run that was interrupted halfway
# saved nothing and simply starts over.
cd "$(dirname "$0")/.." || exit 1
export RAG_LLM_URL=${RAG_LLM_URL:-http://localhost:11434/v1}
for run in a1 b1 c1 d1 a2 b2; do
    out="eval/runs/sampling-$run.json"
    if [ -f "$out" ]; then echo "skip $run (done)"; continue; fi
    echo "===== $run $(date +%H:%M)"
    RAG_LLM_MODEL="qwen3.5-9b-rag-${run:0:1}" uv run eval.py --answers --no-think --temperature model -q --save "$out" \
        | grep -E 'questions, mode|^ALL '
done
echo "all runs done $(date +%H:%M)"
