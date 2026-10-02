#!/bin/bash
#
# Smoke test the V-JEPA 2-AC server: POST a video + hard-coded deterministic
# 7-DOF actions/states, decode the response, and verify the returned
# predicted_hidden tensor shape + that outputs are finite / non-degenerate.
#
# Usage:
#   bash test/vjepa2/video_request_ac.sh <video_path.mp4> [host] [port]
#
# Examples:
#   bash test/vjepa2/video_request_ac.sh test/qwen3-omni/video.webm
#   bash test/vjepa2/video_request_ac.sh clip.mp4 127.0.0.1 20003
#
# What gets sent
# --------------
# The ViT-g AC checkpoint is trained on 64-frame 256×256 clips with tubelet_size=2,
# which gives T_action = 64/2 = 32 per-tubelet timesteps.  Each action + state is
# 7-DOF (``action_embed_dim = 7``).  We build a deterministic linear ramp so:
#   - multiple runs compare byte-identical server-side (for repro)
#   - the ramp is small/smooth enough that the predictor doesn't saturate to NaN
#   - inspecting the client-side stats tells us at a glance if the AC-specific
#     path is firing (non-zero std of predicted_hidden means action/state tokens
#     are actually contributing, not being ignored)
#
# What gets asserted client-side
# ------------------------------
#   * At least one ``modality="video"`` chunk arrives
#   * Chunk size is divisible by hidden_size=1408 (ViT-g embedding dim) — this
#     is what the AC predictor projects back to via ``predictor_proj``
#   * All returned floats are finite (no NaN/Inf — would indicate a load bug)
#   * Non-zero std — zero std would mean the predictor output is constant,
#     which is what we saw with the "uninitialized weights" Phase-1 behavior

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path.mp4> [host] [port]}"
HOST="${2:-127.0.0.1}"
PORT="${3:-20003}"
URL="http://${HOST}:${PORT}/generate"

if [ ! -f "${VIDEO}" ]; then
    echo "Video file not found: ${VIDEO}" >&2
    exit 1
fi

# Build the deterministic actions/states payload in Python so the JSON
# serialization is clean (bash-level numeric arrays are error-prone).  Shape
# is [T_action=32, action_embed_dim=7] for both; use a linear ramp that's
# distinguishable run-to-run if anyone tweaks the seed.
MODEL_KWARGS=$(python3 - <<'PY'
import json

T_ACTION = 32  # num_frames (64) // tubelet_size (2) for facebook/vjepa2-ac-vitg
DOF = 7

# Deterministic ramp: actions walk from -0.5 to 0.5; states from 0.1 to 0.9.
# Broadcast across all 7 DOFs so each timestep's vector is a scalar repeated.
def ramp(lo, hi, T):
    step = (hi - lo) / max(T - 1, 1)
    return [[lo + i * step] * DOF for i in range(T)]

print(json.dumps({
    "actions": ramp(-0.5, 0.5, T_ACTION),
    "states":  ramp( 0.1, 0.9, T_ACTION),
}))
PY
)

TMPFILE=$(mktemp /tmp/vjepa2_ac_response.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo "[vjepa2-ac] POST ${URL}"
echo "  video:         ${VIDEO}"
echo "  model_kwargs:  $(echo "${MODEL_KWARGS}" | head -c 160)..."

curl -sS --fail-with-body -X POST "${URL}" \
    -F "files=@${VIDEO}" \
    -F 'input_modalities=video' \
    -F 'output_modalities=video' \
    -F "model_kwargs=${MODEL_KWARGS}" \
    -o "${TMPFILE}"

python3 - <<PY
import base64
import json
import sys

import numpy as np

HIDDEN_SIZE = 1408  # ViT-g embed_dim (== AC predictor ``predictor_proj.out_features``).

path = "${TMPFILE}"

chunks = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    if msg.get("modality") != "video":
        continue
    data = msg.get("data", "")
    if data:
        chunks.append(base64.b64decode(data))

if not chunks:
    print("No video-modality chunks received.  Raw response:")
    sys.stdout.write(open(path).read())
    sys.exit(1)

raw = b"".join(chunks)
arr = np.frombuffer(raw, dtype=np.float32)
print(f"[vjepa2-ac] received {len(chunks)} chunk(s), {len(raw)} bytes, {arr.size} floats")
print(f"[vjepa2-ac] stats: mean={arr.mean():.4f}  std={arr.std():.4f}  "
      f"min={arr.min():.4f}  max={arr.max():.4f}")

assert np.isfinite(arr).all(), (
    "predicted_hidden contains non-finite values (NaN/Inf) — likely a weight-load bug "
    "or the AC predictor is seeing uninitialized weights."
)

assert arr.size % HIDDEN_SIZE == 0, (
    f"size {arr.size} not divisible by hidden_size={HIDDEN_SIZE}; the AC predictor "
    f"projects back to embed_dim=1408 on output, so something upstream is wrong."
)
tokens = arr.size // HIDDEN_SIZE
print(f"[vjepa2-ac] shape: [{tokens} tokens, {HIDDEN_SIZE} hidden]")

assert arr.std() > 1e-4, (
    f"predicted_hidden has near-zero std ({arr.std():.6g}); this typically means the "
    "predictor weights are still uninitialized (Phase-1 AC behavior) — check the "
    "server logs for an 'uninitialized weights' warning."
)

print("[vjepa2-ac] OK")
PY
