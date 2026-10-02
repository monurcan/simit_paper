"""Step 7: build blinded audit packets for the external (agentic) evaluator.

Every "available" M and L record becomes one packet: {audit_id, benchmark,
image_path, question, answer}. Method identity, reward/advantage values,
selection probabilities, and sibling candidate answers are deliberately
NEVER included -- the evaluator sees exactly one (image, question, answer)
triplet per packet and nothing else, so it cannot infer which method
produced it. The de-anonymizing map (audit_id -> source/method/sample_id)
is kept separately in provenance/audit_manifest.jsonl and is never shown to
the evaluator.

Packets are grouped into one shuffled JSONL per benchmark (eval_packets/
<benchmark>_packets.jsonl) so a single subagent can process one benchmark's
full queue (M + every available method's L) without ever seeing them in a
fixed per-query, per-method order that would leak grouping.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
import hashlib
import json
import os
import random

CACHE_DIR = SIMIT_ROOT + "/analyses/label_noise_model_noise/cache"
PROVENANCE_DIR = SIMIT_ROOT + "/analyses/label_noise_model_noise/provenance"
EVAL_PACKETS_DIR = SIMIT_ROOT + "/analyses/label_noise_model_noise/eval_packets"

SOURCES = [
    ("baseline_answers.jsonl", "M", None),
    ("lfrl_ttrl_selected.jsonl", "L", "ttrl"),
    ("lfrl_empo_selected.jsonl", "L", "empo"),
    ("lfrl_ttrv_selected.jsonl", "L", "ttrv"),
    ("lfrl_srlm_selected.jsonl", "L", "srlm"),
    ("simit_selected.jsonl", "L", "simit"),
]


def audit_id_for(kind, method, benchmark, sample_id):
    key = f"{kind}|{method}|{benchmark}|{sample_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def main():
    manifest = []
    by_benchmark_packets = {}

    for fname, kind, method in SOURCES:
        path = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(path):
            print(f"skip (not found yet): {fname}")
            continue
        rows = [json.loads(l) for l in open(path)]
        n_packets = 0
        for row in rows:
            if kind == "M":
                answer = row["baseline_answer"]
                availability = "available"
            else:
                availability = row["availability"]
                answer = row.get("selected_answer")
            if availability != "available" or answer is None:
                continue

            benchmark = row["benchmark"]
            sample_id = row["sample_id"]
            aid = audit_id_for(kind, method or "baseline", benchmark, sample_id)

            if kind == "M":
                image_path, question = row["image_path"], row["question"]
            elif method == "simit":
                image_path, question = row["selected_image_path"], row["selected_question"]
            else:
                image_path, question = row["image_path"], row["question"]

            manifest.append(
                {
                    "audit_id": aid, "kind": kind, "method": method, "benchmark": benchmark,
                    "sample_id": sample_id, "answer": answer,
                }
            )
            by_benchmark_packets.setdefault(benchmark, []).append(
                {"audit_id": aid, "image_path": image_path, "question": question, "answer": answer}
            )
            n_packets += 1
        print(f"{fname}: {n_packets} available packets")

    os.makedirs(EVAL_PACKETS_DIR, exist_ok=True)
    rng = random.Random("20260913:packet_shuffle")
    total = 0
    for benchmark, packets in by_benchmark_packets.items():
        rng.shuffle(packets)
        out_path = os.path.join(EVAL_PACKETS_DIR, f"{benchmark}_packets.jsonl")
        with open(out_path, "w") as f:
            for p in packets:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        total += len(packets)
        print(f"{benchmark}: {len(packets)} packets -> {out_path}")

    manifest_path = os.path.join(PROVENANCE_DIR, "audit_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nTotal packets: {total} across {len(by_benchmark_packets)} benchmarks")
    print(f"Manifest (NOT for the evaluator): {manifest_path}")


if __name__ == "__main__":
    main()
