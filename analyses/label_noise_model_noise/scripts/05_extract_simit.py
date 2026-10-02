"""Step 6 (per plan): extract SIMIT's retained synthetic supervision for each
selected original query, using the SAME DF+ASC selection logic as the real
pipeline (Bagel/prepare_vqa_data_v2_dfadaptive.py's build_records(), imported
read-only -- never reimplemented). Per the user's explicit instruction, only
DF+ASC-selected samples are used (never round-robin/no-mechanism), with the
best-available DF+ASC hyperparameters for every one of the 17 benchmarks
(see provenance/simit_df_asc_configs.json for where each benchmark's config
came from).

Selection within build_records() is a deterministic keep/truncate, not
weighted, so among a query's retained synthetic candidates we sample ONE
uniformly at random (matching the user's 'uniformly if examples are sampled
uniformly' fallback). If a query retains zero synthetic examples after
DF+ASC filtering, that is recorded as availability="no_retained_supervision"
(never invent an answer).
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, SIMIT_ROOT + "/Bagel")
from prepare_vqa_data_v2_dfadaptive import build_records, SRC_ROOT  # noqa: E402 (read-only import)

from config import CACHE_DIR, PROVENANCE_DIR  # noqa: E402

BASE_SAMPLING_SEED = 20260913


def load_configs():
    return json.load(open(os.path.join(PROVENANCE_DIR, "simit_df_asc_configs.json")))


def main():
    configs = load_configs()
    selected = [json.loads(l) for l in open(os.path.join(CACHE_DIR, "selected_queries.jsonl"))]
    by_bench = {}
    for row in selected:
        by_bench.setdefault(row["benchmark"], set()).add(row["sample_id"])

    out_rows = []
    for bench, wanted_sample_ids in by_bench.items():
        cfg = configs[bench]
        records, stats = build_records(
            bench,
            epsilon=cfg["epsilon"], A0=cfg["A0"], A1=cfg["A1"], B=cfg["B"],
            T_low=cfg["T_low"] if cfg["T_low"] is not None else 0.0,
            T_high=cfg["T_high"] if cfg["T_high"] is not None else 1.0,
            disable_df=cfg["disable_df"], disable_adaptive_k=False,
        )
        print(f"[simit] {bench}: {stats['total_records']} total retained records across "
              f"{stats['n_real_samples']} real queries (config source: {cfg['source']})", flush=True)

        # Group retained records by which original query (sampleN) they belong to.
        by_sample = {}
        for r in records:
            # image path like "ok_vqa_val2014/sample123/synthesized_sample_2.png"
            sample_id = r["image"].split("/")[1]
            by_sample.setdefault(sample_id, []).append(r)

        for sample_id in sorted(wanted_sample_ids, key=lambda s: int(s.replace("sample", ""))):
            retained = by_sample.get(sample_id, [])
            row = {
                "benchmark": bench,
                "sample_id": sample_id,
                "df_asc_config": cfg,
                "n_retained": len(retained),
            }
            if not retained:
                row.update({"availability": "no_retained_supervision", "selected_answer": None,
                            "selected_image_path": None, "selected_question": None,
                            "selection_probability": None})
            else:
                rng = random.Random(f"{BASE_SAMPLING_SEED}:simit:{bench}:{sample_id}")
                chosen = rng.choice(retained)
                question = chosen["conversations"][0]["value"].replace("<image>\n", "", 1)
                answer = chosen["conversations"][1]["value"]
                row.update(
                    {
                        "availability": "available",
                        "selected_image_path": str(SRC_ROOT / chosen["image"]),
                        "selected_question": question,
                        "selected_answer": answer,
                        "selection_probability": 1.0 / len(retained),
                        "retained_image_paths": [r["image"] for r in retained],
                        "sampling_seed": f"{BASE_SAMPLING_SEED}:simit:{bench}:{sample_id}",
                    }
                )
            out_rows.append(row)

    out_path = os.path.join(CACHE_DIR, "simit_selected.jsonl")
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_avail = sum(1 for r in out_rows if r["availability"] == "available")
    print(f"[simit] wrote {len(out_rows)} rows ({n_avail} available) -> {out_path}")


if __name__ == "__main__":
    main()
