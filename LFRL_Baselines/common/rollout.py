from __future__ import annotations

import gc
from typing import List, Tuple

import torch
from bagel.data.data_utils import pil_img2rgb
from bagel.modeling.bagel.qwen2_navit import NaiveCache
from PIL import Image


def _is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _replicate_cache(cache: NaiveCache, n: int) -> NaiveCache:
    """Copies a single-sequence KV cache into n independent, identical copies,
    laid out as n contiguous blocks in the same flat per-layer tensor BAGEL's
    NaiveCache already uses (see qwen2_navit.py) -- this is exactly the layout
    model.prepare_start_tokens(curr_kvlens=[L]*n, ...) already assumes when
    building packed_key_value_indexes for n sequences from scratch, so no
    other bookkeeping needs to change to support n>1 branches from here."""
    new_cache = NaiveCache(cache.num_layers)
    for layer_idx in range(cache.num_layers):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        if k is not None:
            new_cache.key_cache[layer_idx] = k.repeat(n, 1, 1)
            new_cache.value_cache[layer_idx] = v.repeat(n, 1, 1)
    return new_cache


def _decode_text(tokenizer, token_ids: List[int]) -> str:
    """Matches InterleaveInferencer.gen_text's own decode+parse exactly (see
    inferencer.py), so batched decoding is a drop-in, output-equivalent
    replacement for calling gen_text K times.

    The tokenizer's real usable vocab (len(tokenizer)) is smaller than the
    model's lm_head output dimension (padded for hardware alignment), so
    sampling can rarely draw an id in that unused padding range, which
    decode() can't render (confirmed: ids >= len(tokenizer) map to None).
    Treated as a failed/empty completion, same as any other invalid
    generation -- every call site already filters those out."""
    if not token_ids:
        return ""
    try:
        decoded = tokenizer.decode(torch.tensor(token_ids))
    except (TypeError, ValueError, IndexError):
        return ""
    try:
        return decoded.split("<|im_end|>")[0].split("<|im_start|>")[1]
    except IndexError:
        return decoded.strip()


def _decode_batch(
    model,
    prefix_cache: NaiveCache,
    prefix_len: int,
    prefix_pos: int,
    batch_k: int,
    new_token_ids: dict,
    max_new_tokens: int,
    temperature: float,
) -> List[List[int]]:
    """Decodes batch_k parallel sequences from an already-built shared prefix
    cache (prefix_cache is read-only here -- _replicate_cache never mutates
    its source, so the same prefix_cache can be re-passed for a smaller
    batch_k after an OOM retry without rebuilding the (image, prompt) prefix).

    On OOM, returns whatever was generated so far for this batch (matching
    the original single-shot behavior for a genuine mid-decode OOM -- some
    sequences may have real partial content, worth keeping) but re-raises
    the RuntimeError so the caller can tell a partial-progress OOM (worth
    accepting as-is) apart from a total-loss OOM (nothing generated for any
    sequence -- worth retrying at a smaller batch_k instead of discarding
    the whole batch)."""
    eos_id = int(new_token_ids["eos_token_id"])
    generated: List[List[int]] = [[] for _ in range(batch_k)]

    try:
        # _replicate_cache and prepare_start_tokens must be inside this try
        # too -- replicating the prefix cache to batch_k copies is itself a
        # real allocation (this is exactly where a real production OOM hit:
        # ttrv_nocaps_val crashed uncaught here when this was mistakenly
        # left outside the try during the initial refactor).
        past_key_values = _replicate_cache(prefix_cache, batch_k)
        gen_input = model.prepare_start_tokens(
            curr_kvlens=[prefix_len] * batch_k,
            curr_rope=[prefix_pos] * batch_k,
            new_token_ids=new_token_ids,
        )
        key_values_lens = gen_input["key_values_lens"]
        packed_key_value_indexes = gen_input["packed_key_value_indexes"]
        packed_query_position_ids = gen_input["packed_query_position_ids"]
        curr_tokens = gen_input["packed_start_tokens"]

        finished = [False] * batch_k
        extra_inputs = {"mode": "und"} if model.use_moe else {}

        # BAGEL's own generate_text (bagel.py) appends the CURRENT token
        # before sampling the next one, so with max_length=N it runs N
        # forward passes but only keeps N-1 of the sampled tokens -- the
        # very last one is computed and silently discarded when the loop
        # exits. Matching that exactly (max_new_tokens-1 kept tokens),
        # confirmed by a direct greedy-decode comparison against the
        # untouched reference API: without this, this loop kept one
        # extra trailing token the reference never emits.
        for _ in range(max(0, max_new_tokens - 1)):
            packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype
            )
            unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(unpacked)):
                unpacked[i] = unpacked[i] + i
            packed_key_value_indexes = torch.cat(unpacked, dim=0)

            output = model.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            pred_logits = model.language_model.lm_head(output.packed_query_sequence)

            if temperature > 0:
                probs = torch.nn.functional.softmax(pred_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(pred_logits, dim=-1)

            for i in range(batch_k):
                if finished[i]:
                    continue
                tok = int(next_tokens[i].item())
                if tok == eos_id:
                    finished[i] = True
                else:
                    generated[i].append(tok)

            if all(finished):
                break

            unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(unpacked)):
                unpacked[i] = torch.cat(
                    [unpacked[i], torch.tensor([unpacked[i][-1] + 1], device=unpacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(unpacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            curr_tokens = next_tokens
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        # Same leak pattern as train_ttrl.py's packed-training OOM handler:
        # a mid-loop OOM leaves this frame's locals -- especially
        # past_key_values, the single largest tensor here -- bound until
        # the function returns. Drop references before freeing so
        # empty_cache() can actually reclaim them.
        past_key_values = output = pred_logits = probs = None
        next_tokens = curr_tokens = packed_key_value_indexes = None
        key_values_lens = packed_query_position_ids = unpacked = None
        gc.collect()
        torch.cuda.empty_cache()
        raise _BatchOOM(generated) from exc

    return generated


class _BatchOOM(RuntimeError):
    """Carries whatever partial generation happened before an OOM inside
    _decode_batch, so the caller can decide whether to accept it (real
    partial content exists) or retry at a smaller batch_k (nothing
    generated at all -- the case this project's rollout OOM has actually
    been hitting: 4999/5000 steps on a real flickr30k_test run once
    returned all-empty completions because the OOM landed on essentially
    the very first decode iteration, discarding the whole K=32 batch)."""

    def __init__(self, partial: List[List[int]]):
        super().__init__("batched decode OOM")
        self.partial = partial


@torch.no_grad()
def generate_k_completions(
    runtime,
    *,
    image: Image.Image,
    prompt: str,
    k: int,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> List[str]:
    """Samples K completions for one (image, prompt) pair, batched.

    Builds the (image, prompt) prefix's KV cache ONCE via BAGEL's low-level
    inferencer primitives (init_gen_context / update_context_image /
    update_context_text -- the same steps InterleaveInferencer.interleave_inference
    runs internally for understanding_output=True), then decodes all K
    completions TOGETHER in one batched forward pass per token step, instead
    of K sequential single-sequence decode loops.

    This is possible because BAGEL's decode step (Bagel.generate_text in
    bagel.py) is already written in a packed/varlen style generic over any
    number of concurrent sequences -- flash_attn_varlen_func with per-sequence
    cu_seqlens, KV cache as one flat per-layer tensor addressed via a
    per-sequence index map -- with exactly one line hardcoding batch=1 (its
    stopping condition only checks sequence 0's token). This reimplements
    that same loop with a per-sequence EOS mask instead: all K sequences keep
    decoding through max_new_tokens (simplest correct approach -- no dynamic
    batch-shrinking), and tokens generated after a sequence's own EOS are
    simply discarded, matching what early-stopping would have produced.
    BAGEL's own modeling code (shared with the rest of this project) is never
    modified -- this lives entirely in this file, calling only its existing
    public primitives.

    The prompt is passed through unchanged (including its "Question:" prefix):
    the existing eval path does the same, no chat-template wrapping or prefix
    stripping happens anywhere in this call chain.

    This shared HPC node runs other users' jobs concurrently, so a transient
    OOM is expected and not a bug; an OOM building the shared prefix drops the
    whole prompt (returns no completions). An OOM inside the batched decode
    is handled per-batch: if it hit after real tokens were already generated
    for at least one sequence, that partial content is kept (matches the
    original behavior). If it hit with literally nothing generated yet for
    any sequence in the batch -- confirmed via a real job log to be the
    actual dominant failure mode (a full flickr30k_test run once hit this on
    4999/5000 steps, discarding all K completions as empty strings every
    time) -- the batch is retried at half the width instead of being thrown
    away outright, the same chunk-halving pattern already used for packed
    training in train_ttrl.py, just applied to this call site too."""
    was_training = runtime.model.training
    runtime.model.eval()
    model = runtime.model
    new_token_ids = runtime.new_token_ids
    bos_id = int(new_token_ids["bos_token_id"])

    all_generated: List[List[int]] = []
    try:
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            try:
                resized_image = runtime.inferencer.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = runtime.inferencer.init_gen_context()
                gen_context = runtime.inferencer.update_context_image(resized_image, gen_context, vae=False, vit=True)
                gen_context = runtime.inferencer.update_context_text(prompt, gen_context)
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                torch.cuda.empty_cache()
                print("[rollout] OOM building shared prefix, skipping this prompt.", flush=True)
                return ["" for _ in range(k)]

            prefix_len = int(gen_context["kv_lens"][0])
            prefix_pos = int(gen_context["ropes"][0])
            prefix_cache = gen_context["past_key_values"]

            remaining = k
            batch_k = k
            while remaining > 0:
                this_batch = min(batch_k, remaining)
                try:
                    batch_generated = _decode_batch(
                        model, prefix_cache, prefix_len, prefix_pos, this_batch,
                        new_token_ids, max_new_tokens, temperature,
                    )
                    all_generated.extend(batch_generated)
                    remaining -= this_batch
                except _BatchOOM as exc:
                    if any(seq for seq in exc.partial):
                        # Real partial content exists for at least one
                        # sequence -- matches the original "keep whatever
                        # was generated so far" behavior, not worth
                        # discarding by retrying smaller.
                        all_generated.extend(exc.partial)
                        remaining -= this_batch
                        continue
                    if this_batch <= 1:
                        print(
                            f"[rollout] OOM even at batch_k=1, dropping {remaining} completion(s).",
                            flush=True,
                        )
                        all_generated.extend([[] for _ in range(remaining)])
                        remaining = 0
                    else:
                        batch_k = max(1, batch_k // 2)
                        print(
                            f"[rollout] OOM during batched decode with nothing generated, "
                            f"retrying {remaining} completion(s) with batch_k={batch_k}.",
                            flush=True,
                        )
    finally:
        runtime.model.train(was_training)

    return [_decode_text(runtime.tokenizer, [bos_id] + seq) for seq in all_generated]
