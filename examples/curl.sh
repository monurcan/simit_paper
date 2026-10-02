#!/bin/bash
# Quick curl smoke tests against a running mstar server (default port 8000).
set -euo pipefail
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
BASE="http://${HOST}:${PORT}"

echo "== native /generate (works for every model) =="
curl -s "${BASE}/generate" -F 'text=Hello, how are you?'
echo

echo "== OpenAI /v1/models =="
curl -s "${BASE}/v1/models"
echo

echo "== OpenAI /v1/chat/completions (bagel / qwen3_omni) =="
curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"bagel","messages":[{"role":"user","content":"hi"}]}'
echo

echo "== OpenAI /v1/audio/speech (orpheus / qwen3_omni) -> out.wav =="
curl -s "${BASE}/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","input":"hello there","voice":"tara"}' -o out.wav
echo "wrote out.wav"

echo "== OpenAI /v1/images/generations (bagel) =="
curl -s "${BASE}/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"model":"bagel","prompt":"a cat in a hat"}' | head -c 200
echo
