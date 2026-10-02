from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


def _build_attention_mask(split_lens, attn_modes, device: torch.device) -> torch.Tensor:
    sample_len = sum(split_lens)
    allow = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)
    csum = 0
    for seg_len, attn_mode in zip(split_lens, attn_modes):
        seg_slice = slice(csum, csum + seg_len)
        if attn_mode == "causal":
            allow[seg_slice, seg_slice] = torch.ones((seg_len, seg_len), dtype=torch.bool, device=device).tril()
            if csum > 0:
                allow[seg_slice, :csum] = True
        elif attn_mode == "full":
            allow[seg_slice, seg_slice] = True
            if csum > 0:
                allow[seg_slice, :csum] = True
        else:
            raise ValueError(f"Unsupported attn_mode={attn_mode!r}")
        csum += seg_len
    mask = torch.full((sample_len, sample_len), float("-inf"), dtype=torch.float32, device=device)
    mask = mask.masked_fill(allow, 0.0)
    return mask


def _build_branching_attention_mask(
    prefix_split_lens: List[int],
    prefix_attn_modes: List[str],
    branch_lens: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Shared prefix (image, prompt) + K independent branches (completions):
    each branch attends to the full prefix plus itself (causal), but NOT to
    any other branch -- block-diagonal among branches, unlike
    _build_attention_mask's fully-sequential everyone-sees-everyone-before-them
    layout, which would incorrectly let completion 2 see completion 1."""
    prefix_len = sum(prefix_split_lens)
    total_len = prefix_len + sum(branch_lens)
    allow = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)

    csum = 0
    for seg_len, attn_mode in zip(prefix_split_lens, prefix_attn_modes):
        seg_slice = slice(csum, csum + seg_len)
        if attn_mode == "causal":
            allow[seg_slice, seg_slice] = torch.ones((seg_len, seg_len), dtype=torch.bool, device=device).tril()
            if csum > 0:
                allow[seg_slice, :csum] = True
        elif attn_mode == "full":
            allow[seg_slice, seg_slice] = True
            if csum > 0:
                allow[seg_slice, :csum] = True
        else:
            raise ValueError(f"Unsupported attn_mode={attn_mode!r}")
        csum += seg_len

    bcsum = prefix_len
    for blen in branch_lens:
        bslice = slice(bcsum, bcsum + blen)
        allow[bslice, bslice] = torch.ones((blen, blen), dtype=torch.bool, device=device).tril()
        allow[bslice, :prefix_len] = True
        bcsum += blen

    mask = torch.full((total_len, total_len), float("-inf"), dtype=torch.float32, device=device)
    mask = mask.masked_fill(allow, 0.0)
    return mask


def build_understanding_train_batch_multi(
    model,
    tokenizer,
    new_token_ids: Dict[str, int],
    vit_transform,
    *,
    image: Image.Image,
    prompt: str,
    completions: List[str],
    max_prompt_tokens: int = 256,
    max_completion_tokens: int = 256,
) -> Optional[Tuple[Dict, List[int]]]:
    """Batched version of build_understanding_train_batch: packs ALL of the
    given completions for the same (image, prompt) into ONE sequence, sharing
    one image+prompt prefix computation instead of recomputing it per
    completion. Each completion is its own branch in the attention mask (see
    _build_branching_attention_mask) and gets the SAME position-id range
    (right after the shared prompt) since they are parallel alternatives, not
    a continuation of one another. Returns (batch_dict, completion_loss_lens)
    -- the caller splits the returned model(**batch)['ce'] tensor by
    completion_loss_lens (torch.split) to recover each completion's own
    per-token losses and apply its own advantage weight before summing into
    one backward() call."""
    curr_lens = [0]
    curr_rope = [0]

    image_inputs, curr_lens, curr_rope = model.prepare_vit_images(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        images=[image],
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    image_split_len = int(image_inputs["packed_seqlens"][0].item())

    prompt_text = str(prompt or "")
    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_text = tokenizer.decode(prompt_ids[:max_prompt_tokens])

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[prompt_text],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    prompt_split_len = int(prompt_inputs["text_token_lens"][0].item())

    device = prompt_inputs["packed_text_ids"].device
    bos = int(new_token_ids["bos_token_id"])
    eos = int(new_token_ids["eos_token_id"])
    branch_start_pos = int(curr_rope[0])  # same starting position for every branch -- they're parallel, not sequential
    cursor = int(curr_lens[0])  # flat packed-buffer position, grows sequentially across branches

    branch_text_ids, branch_indexes, branch_pos_ids = [], [], []
    branch_ce_indexes, branch_label_ids = [], []
    branch_lens: List[int] = []
    completion_loss_lens: List[int] = []

    for completion in completions:
        completion_ids = tokenizer.encode(str(completion or ""))
        if len(completion_ids) > max_completion_tokens:
            completion_ids = completion_ids[:max_completion_tokens]
        if not completion_ids:
            continue

        shifted = [bos] + completion_ids
        input_ids = shifted + [eos]
        labels = completion_ids + [eos]
        clen = len(input_ids)
        loss_len = len(shifted)

        branch_text_ids.append(torch.tensor(input_ids, dtype=torch.long, device=device))
        branch_indexes.append(torch.arange(cursor, cursor + clen, dtype=torch.long, device=device))
        branch_pos_ids.append(
            torch.arange(branch_start_pos, branch_start_pos + clen, dtype=torch.long, device=device)
        )
        branch_ce_indexes.append(torch.arange(cursor, cursor + loss_len, dtype=torch.long, device=device))
        branch_label_ids.append(torch.tensor(labels, dtype=torch.long, device=device))
        branch_lens.append(clen)
        completion_loss_lens.append(loss_len)
        cursor += clen

    if not branch_lens:
        return None

    packed_text_ids = torch.cat([image_inputs["packed_text_ids"], prompt_inputs["packed_text_ids"], *branch_text_ids])
    packed_text_indexes = torch.cat(
        [image_inputs["packed_text_indexes"], prompt_inputs["packed_text_indexes"], *branch_indexes]
    )
    packed_position_ids = torch.cat(
        [image_inputs["packed_position_ids"], prompt_inputs["packed_text_position_ids"], *branch_pos_ids]
    )
    ce_loss_indexes = torch.cat(branch_ce_indexes)
    packed_label_ids = torch.cat(branch_label_ids)

    sequence_length = cursor
    nested_attention_mask = _build_branching_attention_mask(
        prefix_split_lens=[image_split_len, prompt_split_len],
        prefix_attn_modes=["full", "causal"],
        branch_lens=branch_lens,
        device=device,
    )

    batch = {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
        "nested_attention_masks": [nested_attention_mask],
        "split_lens": [image_split_len, prompt_split_len] + branch_lens,
        "attn_modes": ["full", "causal"] + ["causal"] * len(branch_lens),
        "packed_text_ids": packed_text_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": packed_position_ids,
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": packed_label_ids,
        "packed_vit_tokens": image_inputs["packed_vit_tokens"],
        "packed_vit_token_indexes": image_inputs["packed_vit_token_indexes"],
        "packed_vit_position_ids": image_inputs["packed_vit_position_ids"],
        "vit_token_seqlens": image_inputs["vit_token_seqlens"],
    }
    return batch, completion_loss_lens


def build_understanding_train_batch(
    model,
    tokenizer,
    new_token_ids: Dict[str, int],
    vit_transform,
    *,
    image: Image.Image,
    prompt: str,
    completion: str,
    max_prompt_tokens: int = 256,
    max_completion_tokens: int = 256,
) -> Optional[Dict]:
    """Builds BAGEL's packed/NaViT-format training batch for one (image, prompt,
    completion) triple, with the CE loss restricted to the completion span.
    BAGEL's forward() does not accept plain input_ids/attention_mask -- this
    reconstructs the same packed-tensor convention its own inferencer builds
    internally, ported from the proven self-evolving-uug/Bagel/train reference
    (a different project, same BAGEL packed-format requirement)."""
    curr_lens = [0]
    curr_rope = [0]

    image_inputs, curr_lens, curr_rope = model.prepare_vit_images(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        images=[image],
        transforms=vit_transform,
        new_token_ids=new_token_ids,
    )
    image_split_len = int(image_inputs["packed_seqlens"][0].item())

    prompt_text = str(prompt or "")
    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_text = tokenizer.decode(prompt_ids[:max_prompt_tokens])

    prompt_inputs, curr_lens, curr_rope = model.prepare_prompts(
        curr_kvlens=curr_lens,
        curr_rope=curr_rope,
        prompts=[prompt_text],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )

    completion_ids = tokenizer.encode(str(completion or ""))
    if len(completion_ids) > max_completion_tokens:
        completion_ids = completion_ids[:max_completion_tokens]
    if not completion_ids:
        return None

    bos = int(new_token_ids["bos_token_id"])
    eos = int(new_token_ids["eos_token_id"])

    shifted_completion_ids = [bos] + completion_ids
    completion_input_ids = shifted_completion_ids + [eos]
    completion_labels = completion_ids + [eos]

    completion_start_idx = int(curr_lens[0])
    completion_start_pos = int(curr_rope[0])
    completion_len = len(completion_input_ids)
    completion_loss_len = len(shifted_completion_ids)
    device = prompt_inputs["packed_text_ids"].device

    completion_input_ids_t = torch.tensor(completion_input_ids, dtype=torch.long, device=device)
    completion_indexes_t = torch.arange(
        completion_start_idx, completion_start_idx + completion_len, dtype=torch.long, device=device
    )
    completion_pos_t = torch.arange(
        completion_start_pos, completion_start_pos + completion_len, dtype=torch.long, device=device
    )
    ce_loss_indexes = torch.arange(
        completion_start_idx, completion_start_idx + completion_loss_len, dtype=torch.long, device=device
    )

    prompt_split_len = int(prompt_inputs["text_token_lens"][0].item())

    packed_text_ids = torch.cat(
        [image_inputs["packed_text_ids"], prompt_inputs["packed_text_ids"], completion_input_ids_t]
    )
    packed_text_indexes = torch.cat(
        [image_inputs["packed_text_indexes"], prompt_inputs["packed_text_indexes"], completion_indexes_t]
    )
    packed_position_ids = torch.cat(
        [image_inputs["packed_position_ids"], prompt_inputs["packed_text_position_ids"], completion_pos_t]
    )

    split_lens = [image_split_len, prompt_split_len, completion_len]
    attn_modes = ["full", "causal", "causal"]
    sequence_length = completion_start_idx + completion_len
    nested_attention_mask = _build_attention_mask(split_lens, attn_modes, device)

    return {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
        "nested_attention_masks": [nested_attention_mask],
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "packed_text_ids": packed_text_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_position_ids": packed_position_ids,
        "ce_loss_indexes": ce_loss_indexes,
        "packed_label_ids": torch.tensor(completion_labels, dtype=torch.long, device=device),
        "packed_vit_tokens": image_inputs["packed_vit_tokens"],
        "packed_vit_token_indexes": image_inputs["packed_vit_token_indexes"],
        "packed_vit_position_ids": image_inputs["packed_vit_position_ids"],
        "vit_token_seqlens": image_inputs["vit_token_seqlens"],
    }
