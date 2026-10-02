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
from common.grpo import group_normalized_advantages
from common.packed_batch import build_understanding_train_batch_multi
from common.paths import BAGEL_MODEL_PATH, CHECKPOINT_ROOT
from common.rollout import generate_k_completions
from common.runtime import load_bagel_runtime
from common.voting import lexical_cluster_vote_answer, naive_majority_vote_answer
from common.vqa_scoring import CIDER_SCORED_BENCHMARKS, vqa_soft_accuracy
from ttrl.reward import compute_ttrl_rewards


def parse_args():
    p = argparse.ArgumentParser(description="TTRL: majority-vote pseudo-label + GRPO finetuning of BAGEL-7B (LoRA).")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument(
        "--benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmark names; empty = all 17. TTRL's paper trains one model PER "
        "benchmark independently -- pass a single benchmark here for a paper-faithful run. Multiple "
        "benchmarks pool into one shared LoRA adapter, which is a deliberate deviation, not the default.",
    )
    p.add_argument("--max_samples_per_benchmark", type=int, default=None)
    p.add_argument("--k", type=int, default=8, help="Number of sampled completions per prompt (voting and training).")
    p.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Full passes over the dataset, each with freshly regenerated rollouts and re-computed "
        "majority-vote pseudo-labels from the updated policy (TTRL's 'episodes', paper table: 10/30/80 "
        "for MATH-500/AMC/AIME-2024 by dataset size -- our benchmarks are 500 samples each, matching "
        "MATH-500's size, hence the default of 10).",
    )
    p.add_argument("--voting_mode", type=str, default="naive", choices=["naive", "lexical"])
    p.add_argument("--lexical_similarity_threshold", type=float, default=0.6)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5, help="Peak LR. TTRL's own 5e-7 is for full fine-tuning; LoRA conventionally needs a much higher peak, so this is not a literal copy of the paper's value.")
    p.add_argument("--lr_warmup_ratio", type=float, default=0.03, help="Fraction of total steps spent warming up, matching TTRL's cosine schedule.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--max_image_side",
        type=int,
        default=None,
        help="If set, downscale images so their longer side is at most this many pixels before "
        "encoding (aspect ratio preserved). Off by default (None): most benchmarks' images are "
        "small enough that this never matters. Benchmarks with very high-resolution images (e.g. "
        "nocaps_val/textcaps_val, up to 1024x1024) can push a single generation call's memory use "
        "up by tens of GB, which -- via PyTorch's caching allocator not releasing that reservation "
        "between steps -- saturates an 80GB GPU within a handful of steps on an already-contended "
        "node. Confirmed via direct measurement: one 1024x1024 image at K=32 added 27.5GB; capping "
        "to 512px keeps the incremental cost negligible.",
    )
    p.add_argument(
        "--max_packed_completions",
        type=int,
        default=None,
        help="Cap on completions per packed training forward/backward. Default (None) packs all "
        "of a step's reward-bearing completions in one call, which is fine for most benchmarks "
        "(empirically <1%% of steps ever OOM there). Benchmarks with long completions (verbose "
        "captions) can consistently need more memory than is free at the full K=32 packed size --  "
        "set this lower (e.g. 4) for those so the OOM-retry-with-halving fallback stays a rare "
        "exception instead of firing on most steps.",
    )
    p.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N prompts.")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after this many total prompts across all epochs (for smoke tests).")
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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()] or None
    if benchmarks is None or len(benchmarks) > 1:
        print(
            "[train_ttrl] WARNING: training on "
            f"{'all 17 benchmarks pooled' if benchmarks is None else f'{len(benchmarks)} pooled benchmarks'} "
            "into one shared LoRA adapter. TTRL's paper trains one model PER benchmark independently -- "
            "pass a single --benchmarks value for a paper-faithful run.",
            flush=True,
        )
    base_samples = load_real_samples(
        benchmarks=benchmarks,
        max_samples_per_benchmark=args.max_samples_per_benchmark,
        seed=args.seed,
    )
    print(f"[train_ttrl] {len(base_samples)} prompts/epoch from benchmarks={benchmarks or 'all'}, {args.num_epochs} epoch(s)")

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
    print(f"[train_ttrl] LoRA trainable params: {n_trainable:,}")
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

    run_dir = os.path.join(CHECKPOINT_ROOT, "ttrl", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    start_step = 0
    if args.resume_from is not None:
        loaded_step = load_training_state(args.resume_from, optimizer, scheduler)
        if loaded_step is None:
            print(
                f"[train_ttrl] --resume_from {args.resume_from} has no resume state "
                "(pre-dates this feature) -- starting fresh instead.",
                flush=True,
            )
        else:
            start_step = loaded_step
            print(f"[train_ttrl] resumed from {args.resume_from} at step {start_step}/{total_steps}", flush=True)

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        # Fresh shuffle each epoch; rollouts (and thus majority-vote pseudo-labels)
        # are regenerated from the current, epoch-over-epoch-updated policy below --
        # this is TTRL's actual "episode" mechanism, not a cosmetic reshuffle.
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
            print(f"[train_ttrl] step {step} completions: {completions!r}")
            rewards = compute_ttrl_rewards(
                completions,
                voting_mode=args.voting_mode,
                lexical_similarity_threshold=args.lexical_similarity_threshold,
            )
            advantages = group_normalized_advantages(rewards)

            # Free monitoring metric (GT used for reporting only, never in
            # reward/loss, per the original instruction): score the SAME
            # majority-vote consensus the reward already computed against GT,
            # using the standard VQA soft-accuracy formula. Skipped for
            # CIDEr-scored (captioning) benchmarks -- see vqa_scoring.py.
            monitor_accuracy = None
            if sample.benchmark not in CIDER_SCORED_BENCHMARKS and sample.answers:
                consensus_answer = (
                    naive_majority_vote_answer(completions)
                    if args.voting_mode == "naive"
                    else lexical_cluster_vote_answer(completions, args.lexical_similarity_threshold)
                )
                monitor_accuracy = vqa_soft_accuracy(consensus_answer, sample.answers)

            # visual_gen stays False: we never train the image-generation path.
            model.config.visual_gen = False
            model.config.visual_und = True
            model.train(True)
            optimizer.zero_grad(set_to_none=True)

            n_updated = 0
            updated_pairs = [
                (completion, advantage)
                for completion, advantage in zip(completions, advantages)
                if advantage != 0.0 and completion
            ]
            if updated_pairs:
                # Packed into chunks instead of always one giant forward/backward:
                # for benchmarks with long completions (verbose captions, e.g.
                # flickr30k_test/nocaps_val/textcaps_val), the full K-completion
                # packed batch (dense block-diagonal attention + backward
                # activations) can consistently need more memory than is free --
                # confirmed this is a real capacity limit, not a reference leak
                # (torch.cuda.memory_allocated() stayed flat across repeated OOMs
                # regardless of explicit del + gc.collect()). --max_packed_completions
                # sets a proactive starting chunk size (empirically measured safe
                # sizes: flickr30k_test<=16, nocaps_val<=8, textcaps_val<=2 -- most
                # other benchmarks never OOM at all, so leave it unset for those)
                # so the OOM-retry-with-halving below stays a rare fallback rather
                # than firing on most steps. Gradients from all successful chunks
                # accumulate before one optimizer step.
                remaining = list(updated_pairs)
                chunk_size = len(remaining)
                if args.max_packed_completions is not None:
                    chunk_size = min(chunk_size, args.max_packed_completions)
                while remaining:
                    chunk = remaining[:chunk_size]
                    chunk_completions = [c for c, _ in chunk]
                    chunk_advantages = [a for _, a in chunk]
                    result = batch = outputs = loss = None
                    try:
                        # All completions in this chunk packed into ONE forward/backward
                        # (shared image+prompt prefix, block-diagonal attention among
                        # completions -- see build_understanding_train_batch_multi);
                        # verified against the old one-call-per-completion path to
                        # match within bf16 batching-order noise (~1%, deterministic,
                        # not a masking bug).
                        result = build_understanding_train_batch_multi(
                            model,
                            runtime.tokenizer,
                            runtime.new_token_ids,
                            runtime.vit_transform,
                            image=image,
                            prompt=sample.prompt,
                            completions=chunk_completions,
                            max_completion_tokens=args.max_new_tokens,
                        )
                        if result is not None:
                            batch, completion_loss_lens = result
                            batch = {k: (v.to(runtime.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                                outputs = model(**batch)
                            ce = outputs.get("ce")
                            if ce is not None and ce.numel() > 0:
                                per_completion_ce = ce.split(completion_loss_lens)
                                # Length-normalized REINFORCE with GRPO group-normalized advantage.
                                # ce = -log P(completion) (F.cross_entropy is the NLL), so
                                # log P = -ce, and minimizing -advantage*log P = +advantage*ce
                                # is the correct policy-gradient descent step (increases P for
                                # positive-advantage completions, decreases it for negative --
                                # NOT -advantage*ce, which is backwards). Divide by k (not
                                # n_updated) so a group's effective step size doesn't spike when
                                # only a minority of its samples carry signal, and stays
                                # consistent across chunks regardless of chunk_size.
                                loss = sum(
                                    float(advantage) * ce_i.mean()
                                    for advantage, ce_i in zip(chunk_advantages, per_completion_ce)
                                ) / max(1, args.k)
                                loss.backward()
                                n_updated += len(completion_loss_lens)
                        remaining = remaining[chunk_size:]
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        result = batch = outputs = loss = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        if chunk_size <= 1:
                            print(f"[train_ttrl] step {step}: OOM even at chunk_size=1, dropping one completion.", flush=True)
                            remaining = remaining[1:]
                        else:
                            chunk_size = max(1, chunk_size // 2)
                            print(f"[train_ttrl] step {step}: OOM on packed training batch, retrying {len(remaining)} completions with chunk_size={chunk_size}.", flush=True)

            grad_norm = 0.0
            if n_updated > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm).item())
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            model.train(False)

            elapsed = time.time() - t0
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            log_entry = {
                "step": step,
                "epoch": epoch,
                "benchmark": sample.benchmark,
                "sample_id": sample.sample_id,
                "mean_reward": mean_reward,
                "rewards": rewards,
                "n_updated": n_updated,
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
                "monitor_accuracy": monitor_accuracy,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            monitor_str = f" monitor_acc={monitor_accuracy:.2f}" if monitor_accuracy is not None else ""
            print(
                f"[train_ttrl] step {step}/{total_steps} epoch={epoch}/{args.num_epochs} bench={sample.benchmark} "
                f"mean_reward={mean_reward:.3f} n_updated={n_updated} grad_norm={grad_norm:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s{monitor_str}"
            )

            if step % args.save_every == 0 or step == total_steps:
                ckpt_dir = os.path.join(run_dir, f"step_{step:07d}")
                save_lora_checkpoint(model, ckpt_dir, runtime.lora_config)
                save_training_state(optimizer, scheduler, step, ckpt_dir)
                print(f"[train_ttrl] saved checkpoint -> {ckpt_dir}")
        else:
            continue
        break

    print("[train_ttrl] done.")


if __name__ == "__main__":
    main()
