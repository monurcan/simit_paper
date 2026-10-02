"""Step 2: build blinded audit packets for the external (agentic) evaluator.

Every row from cache/<setting>_selected.jsonl becomes one packet:
{audit_id, image_path, question, answer}. Rows are already the pool-sampled
triplets (see 01_extract_settings.py), so no availability filtering is
needed here. audit_id is keyed on the candidate's own image path (not just
setting/benchmark/sample_id), since pooled sampling can pick more than one
candidate from the same query. The setting name is
deliberately NEVER included -- the evaluator sees exactly one (image,
question, answer) triplet per packet and nothing else, so it cannot infer
which pipeline variant produced it. The de-anonymizing map (audit_id ->
setting/benchmark/sample_id) is kept separately in
provenance/audit_manifest.jsonl and is never shown to the evaluator.

Packets are grouped into one shuffled JSONL per benchmark (eval_packets/
<benchmark>_packets.jsonl) so a single subagent can process one benchmark's
full queue (all 6 settings' available triplets for that benchmark's 10
queries) without seeing them in a fixed per-setting order that would leak
grouping.
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import BENCHMARKS, CACHE_DIR, EVAL_PACKETS_DIR, PROVENANCE_DIR, SETTINGS


def audit_id_for(setting, benchmark, sample_id, image_rel_path):
    key = f"{setting}|{benchmark}|{sample_id}|{image_rel_path}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def main():
    manifest = []
    by_benchmark_packets = {b: [] for b in BENCHMARKS}

    for setting_name in SETTINGS:
        path = os.path.join(CACHE_DIR, f"{setting_name}_selected.jsonl")
        rows = [json.loads(l) for l in open(path)]
        n_packets = 0
        for row in rows:
            benchmark, sample_id = row["benchmark"], row["sample_id"]
            aid = audit_id_for(setting_name, benchmark, sample_id, row["image_rel_path"])
            manifest.append({
                "audit_id": aid, "setting": setting_name, "benchmark": benchmark,
                "sample_id": sample_id, "answer": row["answer"],
            })
            by_benchmark_packets[benchmark].append({
                "audit_id": aid, "image_path": row["image_path"],
                "question": row["question"], "answer": row["answer"],
            })
            n_packets += 1
        print(f"{setting_name}: {n_packets} available packets")

    rng = random.Random("20260915:packet_shuffle")
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
