"""Wait for the standalone benchmark mstar server to be genuinely ready.

Mirrors the project's established pattern (similar_samples_llms_eval_lite/
synthesize_vqa.py's _restart_server): /health responds before the model has
actually finished loading, so a real warmup chat() call with a long timeout
is required to confirm true readiness, not just the /health endpoint.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import sys
import time
import urllib.request

sys.path.insert(0, SIMIT_ROOT)
from mstar import MStarClient

SERVER_URL = "http://localhost:8010"


def wait_ready(max_health_wait_s: int = 600) -> MStarClient:
    deadline = time.time() + max_health_wait_s
    healthy = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            healthy = True
            break
        except Exception:
            time.sleep(5)
    if not healthy:
        raise RuntimeError("server /health never responded")

    print("[wait_ready] /health OK, doing warmup chat (may block ~10min on cold weight load)...")
    client = MStarClient(SERVER_URL, timeout=600.0)
    t0 = time.perf_counter()
    client.chat("Hi.", think_mode=False, max_output_tokens=5)
    print(f"[wait_ready] warmup chat completed in {time.perf_counter() - t0:.1f}s — server truly ready")
    return client


if __name__ == "__main__":
    wait_ready()
    print("READY")
