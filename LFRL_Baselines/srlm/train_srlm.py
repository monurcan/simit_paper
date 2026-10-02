from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.checkpoint import load_training_state, save_lora_checkpoint, save_training_state
from common.dataset import load_real_samples
from common.lora_utils import disabled_adapter
from common.packed_batch import build_understanding_train_batch_multi
from common.paths import BAGEL_MODEL_PATH, CHECKPOINT_ROOT
from common.rollout import generate_k_completions
from common.runtime import load_bagel_runtime
from srlm.judge import score_completions, select_preference_pair


def parse_args():
    p = argparse.ArgumentParser(
        description="SRLM: self-rewarding (LLM-as-a-Judge) preference pairs + DPO finetuning of BAGEL-7B (LoRA), "
        "following MM-UPT's own SRLM baseline recipe applied to this project's fixed benchmark prompts."
    )
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--benchmarks", type=str, default="")
    p.add_argument("--max_samples_per_benchmark", type=int, default=None)
    p.add_argument("--k", type=int, default=16, help="Candidate completions per prompt, judged to pick a best/worst DPO pair.")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--judge_max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0, help="Candidate-generation sampling temperature.")
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature (matches the reference implementation's default).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_image_side", type=int, default=None)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume_from", type=str, default=None,
        help="Checkpoint dir (a step_XXXXXXX/ folder) to resume from. Loads LoRA weights, "
        "optimizer/scheduler state, and the step counter. If the checkpoint predates the "
        "resume feature (no optimizer.pt/scheduler.pt/training_state.json), falls back to "
        "starting fresh -- not an error.",
    )
    return p.parse_args()


def _completion_logprobs(model, tokenizer, new_token_ids, vit_transform, image, prompt, completions, max_new_tokens):
    """Packs [chosen, rejected] into one forward pass and returns their mean
    (length-normalized) log-probs, matching the reference DPO implementation's
    masked-mean convention (github.com/lucidrains/self-rewarding-lm-pytorch,
    dpo.py). Returns None on OOM or if a completion has zero encodable
    tokens (build_understanding_train_batch_multi already drops those)."""
    result = build_understanding_train_batch_multi(
        model,
        tokenizer,
        new_token_ids,
        vit_transform,
        image=image,
        prompt=prompt,
        completions=completions,
        max_completion_tokens=max_new_tokens,
    )
    if result is None:
        return None
    batch, completion_loss_lens = result
    if len(completion_loss_lens) != len(completions):
        return None
    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        outputs = model(**batch)
    ce = outputs.get("ce")
    if ce is None or ce.numel() == 0:
        return None
    per_completion_ce = ce.split(completion_loss_lens)
    # ce is -log P per token (cross-entropy); mean log-prob per completion = -mean(ce).
    return [-ce_i.mean() for ce_i in per_completion_ce]


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()] or None
    if benchmarks is None or len(benchmarks) > 1:
        print(
            "[train_srlm] WARNING: training on "
            f"{'all 17 benchmarks pooled' if benchmarks is None else f'{len(benchmarks)} pooled benchmarks'} "
            "into one shared LoRA adapter.",
            flush=True,
        )
    base_samples = load_real_samples(
        benchmarks=benchmarks,
        max_samples_per_benchmark=args.max_samples_per_benchmark,
        seed=args.seed,
    )
    print(f"[train_srlm] {len(base_samples)} prompts/epoch from benchmarks={benchmarks or 'all'}, {args.num_epochs} epoch(s)")

    runtime = load_bagel_runtime(
        model_path=BAGEL_MODEL_PATH,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_checkpoint_path=args.resume_from,
    )
    model = runtime.model

    trainable_params = [p for p in model.language_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[train_srlm] LoRA trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = args.num_epochs * len(base_samples)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(round(total_steps * args.lr_warmup_ratio))),
        num_training_steps=max(1, total_steps),
    )

    run_dir = os.path.join(CHECKPOINT_ROOT, "srlm", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    start_step = 0
    if args.resume_from is not None:
        loaded_step = load_training_state(args.resume_from, optimizer, scheduler)
        if loaded_step is None:
            print(
                f"[train_srlm] --resume_from {args.resume_from} has no resume state "
                "(pre-dates this feature) -- starting fresh instead.",
                flush=True,
            )
        else:
            start_step = loaded_step
            print(f"[train_srlm] resumed from {args.resume_from} at step {start_step}/{total_steps}", flush=True)

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_samples = list(base_samples)
        random.Random(args.seed + epoch).shuffle(epoch_samples)

        for sample in epoch_samples:
            step += 1
            if args.max_steps is not None and step > args.max_steps:
                break
            if step <= start_step:
                continue
            t0 = time.time()
            image = Image.open(sample.image_path).convert("RGB")
            if args.max_image_side is not None:
                w, h = image.size
                scale = args.max_image_side / max(w, h)
                if scale < 1.0:
                    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))

            completions = generate_k_completions(
                runtime,
                image=image,
                prompt=sample.prompt,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            scores = score_completions(
                runtime,
                image=image,
                instruction=sample.prompt,
                completions=completions,
                max_new_tokens=args.judge_max_new_tokens,
            )
            pair = select_preference_pair(completions, scores)

            n_updated = 0
            loss_value = None
            if pair is not None:
                chosen, rejected, chosen_score, rejected_score = pair
                model.config.visual_gen = False
                model.config.visual_und = True
                model.train(True)
                optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.no_grad(), disabled_adapter(model):
                        ref_logprobs = _completion_logprobs(
                            model, runtime.tokenizer, runtime.new_token_ids, runtime.vit_transform,
                            image, sample.prompt, [chosen, rejected], args.max_new_tokens,
                        )
                    policy_logprobs = _completion_logprobs(
                        model, runtime.tokenizer, runtime.new_token_ids, runtime.vit_transform,
                        image, sample.prompt, [chosen, rejected], args.max_new_tokens,
                    )
                    if ref_logprobs is not None and policy_logprobs is not None:
                        ref_chosen, ref_rejected = ref_logprobs
                        policy_chosen, policy_rejected = policy_logprobs
                        policy_logratios = policy_chosen - policy_rejected
                        ref_logratios = ref_chosen.detach() - ref_rejected.detach()
                        loss = -torch.nn.functional.logsigmoid(args.beta * (policy_logratios - ref_logratios))
                        loss.backward()
                        loss_value = float(loss.item())
                        n_updated = 1
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[train_srlm] step {step}: OOM computing DPO loss, skipping this step.", flush=True)

            grad_norm = 0.0
            if n_updated > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm).item())
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            model.train(False)

            elapsed = time.time() - t0
            valid_scores = [s for s in scores if s is not None]
            log_entry = {
                "step": step,
                "epoch": epoch,
                "benchmark": sample.benchmark,
                "sample_id": sample.sample_id,
                "n_candidates": len(completions),
                "n_scored": len(valid_scores),
                "mean_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
                "pair_found": pair is not None,
                "loss": loss_value,
                "n_updated": n_updated,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            loss_str = f" loss={loss_value:.4f}" if loss_value is not None else " loss=n/a"
            print(
                f"[train_srlm] step {step}/{total_steps} epoch={epoch}/{args.num_epochs} bench={sample.benchmark} "
                f"n_scored={len(valid_scores)}/{len(completions)} pair_found={pair is not None}{loss_str} "
                f"n_updated={n_updated} grad_norm={grad_norm:.4f} lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s"
            )

            if step % args.save_every == 0 or step == total_steps:
                ckpt_dir = os.path.join(run_dir, f"step_{step:07d}")
                save_lora_checkpoint(model, ckpt_dir, runtime.lora_config)
                save_training_state(optimizer, scheduler, step, ckpt_dir)
                print(f"[train_srlm] saved checkpoint -> {ckpt_dir}")
        else:
            continue
        break

    print("[train_srlm] done.")


if __name__ == "__main__":
    main()
