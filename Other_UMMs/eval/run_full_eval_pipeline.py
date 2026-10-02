"""SIMIT-ICL evaluation pipeline for ONE benchmark with SenseNova-U1 or Lance (paper App. K).

Runs, in one process (one model load, reused throughout):
  1. Confidence computation for every eval query (<root>/<bench>) and validation
     query (<root>/<bench>_val50) and their synthesized samples that lack one
     (same logic as similar_samples_llms_eval_lite/compute_confidence.py).
  2. Zero-shot baseline (K=0) on the 500-sample eval split.
  3. Optuna search of the DF + ABA parameters on the 50-sample validation split
     (<bench>_lite_val50), same search space as the BAGEL sweeps (App. G.1).
  4. The best validation configuration, evaluated once on the eval split.

The eval split is never used to choose hyperparameters.
"""
import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

TASK_METRIC = {
    "ai2d": "exact_match", "chartqa": "relaxed_overall", "coco2017_cap_val": "coco_CIDEr",
    "docvqa_val": "anls", "flickr30k_test": "flickr_CIDEr", "gqa": "exact_match",
    "infovqa_val": "anls", "mmbench_cn_dev": "gpt_eval_score", "mmbench_en_dev": "gpt_eval_score",
    "nocaps_val": "nocaps_CIDEr", "ok_vqa_val2014": "exact_match", "refcoco_bbox_val": "refcoco_CIDEr",
    "seedbench": "seed_all", "textcaps_val": "textcaps_CIDEr", "textvqa_val": "exact_match",
    "vizwiz_vqa_val": "exact_match", "vqav2_val": "exact_match",
}

MAX_NEW_TOKENS = {
    "ai2d": 16, "chartqa": 16, "coco2017_cap_val": 64, "docvqa_val": 32, "flickr30k_test": 64,
    "gqa": 16, "infovqa_val": 32, "mmbench_cn_dev": 256, "mmbench_en_dev": 1024, "nocaps_val": 64,
    "ok_vqa_val2014": 16, "refcoco_bbox_val": 64, "seedbench": 16, "textcaps_val": 64,
    "textvqa_val": 16, "vizwiz_vqa_val": 16, "vqav2_val": 16,
}
DEFAULT_MAX_NEW_TOKENS = 64
QUESTION_PREFIX_RE = re.compile(r"(^|\n)Question:\s*")


def strip_question_prefix(text: str) -> str:
    return QUESTION_PREFIX_RE.sub(r"\1", text)


def list_synthetic_indices(sample_dir: Path):
    indices = []
    for p in sample_dir.glob("synthesized_sample_*_answers.txt"):
        m = re.match(r"synthesized_sample_(\d+)_answers\.txt", p.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def compute_confidence_for_benchmark(client, bench_dir: Path, max_new_tokens: int):
    n_real, n_synth = 0, 0
    for sample_dir in sorted(bench_dir.iterdir(), key=lambda p: p.name):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
            continue

        real_out = sample_dir / "original_sample_confidence.json"
        if not real_out.exists():
            image_path = sample_dir / "original_sample.png"
            prompt_path = sample_dir / "original_sample_prompt.txt"
            if image_path.exists() and prompt_path.exists():
                prompt_text = strip_question_prefix(prompt_path.read_text().strip())
                image = Image.open(image_path).convert("RGB")
                record = client.score_greedy(image, prompt_text, max_new_tokens)
                real_out.write_text(json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2))
                n_real += 1

        for idx in list_synthetic_indices(sample_dir):
            synth_out = sample_dir / f"synthesized_sample_{idx}_confidence.json"
            if synth_out.exists():
                continue
            image_path = sample_dir / f"synthesized_sample_{idx}.png"
            prompt_path = sample_dir / f"synthesized_sample_{idx}_prompt.txt"
            answers_path = sample_dir / f"synthesized_sample_{idx}_answers.txt"
            if not (image_path.exists() and prompt_path.exists() and answers_path.exists()):
                continue
            prompt_text = prompt_path.read_text().strip()
            answer_text = answers_path.read_text().strip()
            image = Image.open(image_path).convert("RGB")
            record = client.score_teacher_forced(image, prompt_text, answer_text)
            synth_out.write_text(json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2))
            n_synth += 1

        if (n_real + n_synth) % 100 == 0 and (n_real + n_synth) > 0:
            print(f"[confidence] real={n_real} synth={n_synth}", flush=True)

    print(f"[confidence] DONE real={n_real} synth={n_synth}", flush=True)


def get_metric_value(results_dict, metric_key):
    for key in (metric_key, f"{metric_key},none"):
        if key in results_dict:
            return results_dict[key]
    for key, value in results_dict.items():
        if key.startswith(f"{metric_key},"):
            return value
    return None


def evaluate(model, task, limit, metric_key):
    from lmms_eval.evaluator import simple_evaluate
    res = simple_evaluate(model=model, tasks=[task], limit=limit, log_samples=False)
    return get_metric_value(res["results"][task], metric_key)


def apply_dfasc_params(model, params, num_generations):
    from eval.icl_retriever import _compute_tau
    model.num_icl_examples = params["num_icl_examples"]
    model.retriever.df_enabled = True
    model.retriever.asc_enabled = True
    model.retriever.epsilon = params["epsilon"]
    model.retriever.A0 = params["A0"]
    model.retriever.A1 = params["A1"]
    model.retriever.B = params["B"]
    model.retriever.T_low = params["T_low"]
    model.retriever.T_high = params["T_high"]
    model.retriever._tau = _compute_tau(params["epsilon"], params["A0"], params["A1"], params["B"], num_generations)


def load_cached_results(out_path: Path):
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_results(out_path: Path, results: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="benchmark dir name, e.g. ai2d")
    ap.add_argument("--icl-samples-root", required=True)
    ap.add_argument("--num-generations", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=50, help="Optuna trials on the validation split")
    ap.add_argument("--optuna-dir", default=None,
                     help="dir for per-benchmark persistent Optuna sqlite DBs (default: <out>'s parent /optuna)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sensenova", choices=["sensenova", "lance"])
    args = ap.parse_args()

    bench = args.bench
    task, val_task = f"{bench}_lite", f"{bench}_lite_val50"
    metric_key = TASK_METRIC[bench]
    max_new_tokens = MAX_NEW_TOKENS.get(bench, DEFAULT_MAX_NEW_TOKENS)
    out_path = Path(args.out)
    optuna_dir = Path(args.optuna_dir) if args.optuna_dir else out_path.parent / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    results = load_cached_results(out_path)
    results.update(bench=bench, metric=metric_key)

    from eval.sensenova_icl_model import LanceICL, SensenovaICL

    model_cls = LanceICL if args.model == "lance" else SensenovaICL
    model = model_cls(icl_samples_root=args.icl_samples_root, num_icl_examples=0, num_generations=args.num_generations)

    print(f"=== [{bench}] step 1: confidence computation ===", flush=True)
    for folder in (bench, f"{bench}_val50"):
        compute_confidence_for_benchmark(model.client, Path(args.icl_samples_root) / folder, max_new_tokens)

    if results.get("zero_shot") is None:
        print(f"\n=== [{bench}] step 2: zero-shot baseline (K=0) on the eval split ===", flush=True)
        model.num_icl_examples = 0
        model.retriever.df_enabled = False
        model.retriever.asc_enabled = False
        results["zero_shot"] = evaluate(model, task, None, metric_key)
        write_results(out_path, results)
    print(f"[{bench}] zero-shot {metric_key}={results['zero_shot']}", flush=True)

    print(f"\n=== [{bench}] step 3: DF+ABA search on the validation split ({args.n_trials} trials) ===", flush=True)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{optuna_dir / (bench + '_simit_icl_val50.db')}"
    study = optuna.create_study(study_name=f"{bench}_simit_icl_val50", storage=storage, load_if_exists=True,
                                direction="maximize")
    icl_choices = [c for c in [2, 4, 6, 8] if c <= args.num_generations] or [args.num_generations]

    def objective(trial):
        params = dict(epsilon=trial.suggest_float("epsilon", 0.0, 0.5), A0=trial.suggest_float("A0", 0.05, 0.6))
        params["A1"] = trial.suggest_float("A1", params["A0"], 0.9)
        params["B"] = trial.suggest_float("B", 0.0, 0.5)
        params["T_low"] = trial.suggest_float("T_low", 0.0, 0.5)
        params["T_high"] = trial.suggest_float("T_high", params["T_low"], 1.0)
        params["num_icl_examples"] = trial.suggest_categorical("num_icl_examples", icl_choices)
        apply_dfasc_params(model, params, args.num_generations)
        score = evaluate(model, val_task, None, metric_key)
        print(f"[{bench}] trial {trial.number} val {metric_key}={score} params={params}", flush=True)
        return score if score is not None else float("-inf")

    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    best = study.best_trial
    results["best_val"] = {"score": best.value, "params": best.params, "trial": best.number}
    if results.get("simit_icl", {}).get("params") != best.params:
        print(f"\n=== [{bench}] step 4: best validation config on the eval split ===", flush=True)
        apply_dfasc_params(model, best.params, args.num_generations)
        results["simit_icl"] = {"score": evaluate(model, task, None, metric_key), "params": best.params}
        write_results(out_path, results)
    print(json.dumps(results, indent=2, default=str))
    write_results(out_path, results)


if __name__ == "__main__":
    main()
