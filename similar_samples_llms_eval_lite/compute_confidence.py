"""
Compute BAGEL model confidence (per-token log-probabilities) for real and
synthesized VQA samples produced by the similar-sample generation pipeline.

Real samples: greedy-decode an answer (no GT used), score the generated
tokens under the model's own predicted distribution (trivial under greedy:
log of the max-prob token at each step).

Synthesized samples: teacher-force the saved GT answer in a single forward
pass and score it via standard shifted next-token log-probabilities.

Does NOT use mstar / lmms-eval's generate_until path (that only returns
text, no scores). Talks to BAGEL directly via its own low-level packed
generation primitives, mirroring Bagel/inferencer.py's InterleaveInferencer
and Bagel/modeling/bagel/bagel.py's generate_text loop.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

BAGEL_ROOT = SIMIT_ROOT + "/Bagel"
sys.path.insert(0, BAGEL_ROOT)

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch  # noqa: E402
from data.data_utils import add_special_tokens, pil_img2rgb  # noqa: E402
from data.transforms import ImageTransform  # noqa: E402
from inferencer import InterleaveInferencer  # noqa: E402
from modeling.autoencoder import load_ae  # noqa: E402
from modeling.bagel import (  # noqa: E402
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402

MODEL_PATH = SIMIT_ROOT + "/Bagel/models/BAGEL-7B-MoT"

# Per-benchmark max_new_tokens, matching lmms-eval's generation_kwargs.max_new_tokens
# where the task yaml defines one explicitly. A handful of benchmarks in this project
# (ok_vqa_val2014, refcoco_bbox_val, seedbench, textcaps_val, textvqa_val, vizwiz_vqa_val,
# nocaps_val, mathvista) only specify an "until" stop-string in their yaml (meant for
# chat-template models) and no explicit max_new_tokens -- BAGEL has no such stop-string
# mechanism (it stops on its own EOS token), so for those we fall back to a benchmark-shaped
# default (short single-word/phrase answers -> 16, captions -> 64, open-ended math -> 512).
MAX_NEW_TOKENS = {
    "ai2d": 16,
    "chartqa": 16,
    "coco2017_cap_val": 64,
    "docvqa_val": 32,
    "flickr30k_test": 64,
    "gqa": 16,
    "infovqa_val": 32,
    "mathvista": 512,
    "mmbench_cn_dev": 256,
    "mmbench_en_dev": 1024,
    "nocaps_val": 64,
    "ok_vqa_val2014": 16,
    "refcoco_bbox_val": 64,
    "seedbench": 16,
    "textcaps_val": 64,
    "textvqa_val": 16,
    "vizwiz_vqa_val": 16,
    "vqav2_val": 16,
}
DEFAULT_MAX_NEW_TOKENS = 64

QUESTION_PREFIX_RE = re.compile(r"(^|\n)Question:\s*")


def strip_question_prefix(text: str) -> str:
    return QUESTION_PREFIX_RE.sub(r"\1", text)


def load_model():
    model_path = MODEL_PATH

    llm_config = Qwen2Config.from_json_file(f"{model_path}/llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(f"{model_path}/vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=f"{model_path}/ae.safetensors")

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    checkpoint = f"{model_path}/ema.safetensors"
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def _build_prompt_context(inferencer, image, prompt_text):
    model = inferencer.model
    gen_context = inferencer.init_gen_context()
    img = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
    gen_context = inferencer.update_context_image(img, gen_context, vae=False, vit=True)
    gen_context = inferencer.update_context_text(prompt_text, gen_context)
    return gen_context


@torch.no_grad()
def generate_and_score_real(inferencer, image, prompt_text, max_new_tokens):
    """Greedy-decode an answer with no GT, scoring each generated token under
    the model's own predicted distribution (log of the argmax prob, by
    construction of greedy decoding). Mirrors Bagel.generate_text's loop."""
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    end_token_id = new_token_ids["eos_token_id"]
    extra_inputs = {"mode": "und"} if model.use_moe else {}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        gen_context = _build_prompt_context(inferencer, image, prompt_text)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        start = model.prepare_start_tokens(kv_lens, ropes, new_token_ids)
        packed_key_value_indexes = start["packed_key_value_indexes"]
        key_values_lens = start["key_values_lens"]
        packed_query_position_ids = start["packed_query_position_ids"]
        curr_tokens = start["packed_start_tokens"]

        generated_sequence = []
        all_logprobs = []
        step = 0
        while step < max_new_tokens:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = model.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype
            )
            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = uppacked[i] + i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

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
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            curr_tokens = torch.argmax(pred_logits, dim=-1)
            all_logprobs.append(log_probs[0, curr_tokens[0]].item())

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if curr_tokens[0] == end_token_id:
                break

    answer_token_ids = [t.item() for t in generated_sequence[1:]]
    answer_logprobs = all_logprobs[: len(answer_token_ids)]
    answer_text = tokenizer.decode(answer_token_ids).strip() if answer_token_ids else ""
    return answer_text, answer_token_ids, answer_logprobs


@torch.no_grad()
def score_synthetic_teacher_forced(inferencer, image, prompt_text, answer_text):
    """Single forward pass: teacher-force the GT answer text after the
    image+question context and score it with standard shifted next-token
    log-probabilities (position i-1's logits score token i)."""
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    extra_inputs = {"mode": "und"} if model.use_moe else {}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        gen_context = _build_prompt_context(inferencer, image, prompt_text)
        past_key_values = gen_context["past_key_values"]
        curr_kvlen = gen_context["kv_lens"][0]
        curr_position_id = gen_context["ropes"][0]

        answer_token_ids = tokenizer.encode(answer_text)
        if len(answer_token_ids) == 0:
            return [], []

        bos = new_token_ids["bos_token_id"]
        text_ids = [bos] + answer_token_ids  # start-of-turn trigger + GT tokens, no EOS appended

        packed_text_ids = torch.tensor(text_ids, dtype=torch.long)
        packed_text_position_ids = torch.tensor(
            list(range(curr_position_id, curr_position_id + len(text_ids))), dtype=torch.long
        )
        packed_key_value_indexes = torch.tensor(list(range(0, curr_kvlen)), dtype=torch.long)
        packed_text_indexes = torch.tensor(
            list(range(curr_kvlen, curr_kvlen + len(text_ids))), dtype=torch.long
        )
        text_token_lens = torch.tensor([len(text_ids)], dtype=torch.int)
        key_values_lens = torch.tensor([curr_kvlen], dtype=torch.int)

        packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)

        output = model.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=True,
            **extra_inputs,
        )
        pred_logits = model.language_model.lm_head(output.packed_query_sequence)
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)

    # position i in text_ids predicts text_ids[i+1]; text_ids[0]=bos trigger,
    # text_ids[1:]=answer_token_ids, so log_probs[i] scores answer_token_ids[i].
    answer_logprobs = [log_probs[i, tok_id].item() for i, tok_id in enumerate(answer_token_ids)]
    return answer_token_ids, answer_logprobs


def build_confidence_record(answer_text, token_ids, logprobs, tokenizer):
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    probs = [float(torch.tensor(lp).exp()) for lp in logprobs]
    record = {
        "answer_text": answer_text,
        "answer_tokens_str": [{"id": tid, "str": s} for tid, s in zip(token_ids, token_strs)],
        "token_logprobs": logprobs,
        "token_probs": probs,
        "num_answer_tokens": len(token_ids),
    }
    if token_ids:
        avg_logprob = sum(logprobs) / len(logprobs)
        record["avg_answer_logprob"] = avg_logprob
        record["avg_answer_prob"] = float(torch.tensor(avg_logprob).exp())
        min_logprob = min(logprobs)
        record["min_token_logprob"] = min_logprob
        record["min_token_prob"] = float(torch.tensor(min_logprob).exp())
    else:
        record["avg_answer_logprob"] = None
        record["avg_answer_prob"] = None
        record["min_token_logprob"] = None
        record["min_token_prob"] = None
    return record


def discover_samples(base_dir: Path, benchmarks=None):
    for bench_dir in sorted(base_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        if benchmarks and bench_dir.name not in benchmarks:
            continue
        for sample_dir in sorted(bench_dir.iterdir(), key=lambda p: p.name):
            if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
                continue
            yield bench_dir.name, sample_dir


def list_synthetic_indices(sample_dir: Path):
    indices = []
    for p in sample_dir.glob("synthesized_sample_*_answers.txt"):
        m = re.match(r"synthesized_sample_(\d+)_answers\.txt", p.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def process_real_sample(inferencer, bench_name, sample_dir, overwrite=False, debug=False):
    out_path = sample_dir / "original_sample_confidence.json"
    if out_path.exists() and not overwrite and not debug:
        return None

    image_path = sample_dir / "original_sample.png"
    prompt_path = sample_dir / "original_sample_prompt.txt"
    if not image_path.exists() or not prompt_path.exists():
        return None

    raw_prompt = prompt_path.read_text().strip()
    prompt_text = strip_question_prefix(raw_prompt)
    # validation folders are named "<bench>_val50" and decode exactly like their benchmark
    max_new_tokens = MAX_NEW_TOKENS.get(bench_name.removesuffix("_val50"), DEFAULT_MAX_NEW_TOKENS)

    image = Image.open(image_path)
    answer_text, token_ids, logprobs = generate_and_score_real(inferencer, image, prompt_text, max_new_tokens)
    record = build_confidence_record(answer_text, token_ids, logprobs, inferencer.tokenizer)

    if debug:
        return {"prompt": prompt_text, "record": record}

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def process_synthetic_sample(inferencer, bench_name, sample_dir, idx, overwrite=False, debug=False):
    out_path = sample_dir / f"synthesized_sample_{idx}_confidence.json"
    if out_path.exists() and not overwrite and not debug:
        return None

    image_path = sample_dir / f"synthesized_sample_{idx}.png"
    prompt_path = sample_dir / f"synthesized_sample_{idx}_prompt.txt"
    answers_path = sample_dir / f"synthesized_sample_{idx}_answers.txt"
    if not image_path.exists() or not prompt_path.exists() or not answers_path.exists():
        return None

    prompt_text = prompt_path.read_text().strip()
    answer_text = answers_path.read_text().strip()

    image = Image.open(image_path)
    token_ids, logprobs = score_synthetic_teacher_forced(inferencer, image, prompt_text, answer_text)
    record = build_confidence_record(answer_text, token_ids, logprobs, inferencer.tokenizer)

    if debug:
        return {"prompt": prompt_text, "record": record}

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def print_debug_result(kind, bench_name, sample_dir, idx, result):
    print(f"\n=== {kind} | {bench_name} | {sample_dir.name}" + (f" | synth idx {idx}" if idx is not None else "") + " ===")
    print(f"--- prompt ---\n{result['prompt']}")
    rec = result["record"]
    print(f"--- answer_text ---\n{rec['answer_text']!r}")
    print(f"--- per-token (str, logprob, prob) ({rec['num_answer_tokens']} tokens) ---")
    for t, lp, p in zip(rec["answer_tokens_str"], rec["token_logprobs"], rec["token_probs"]):
        print(f"  {t['str']!r:20s} logprob={lp:.4f}  prob={p:.4f}")
    print(f"avg_answer_prob={rec['avg_answer_prob']}  min_token_prob={rec['min_token_prob']}")


def run_debug(datasets, num_debug, seed):
    inferencer = load_model()
    rng = random.Random(seed)

    all_real = []
    all_synth = []
    for base in datasets:
        base = Path(base)
        for bench_name, sample_dir in discover_samples(base):
            all_real.append((bench_name, sample_dir))
            for idx in list_synthetic_indices(sample_dir):
                all_synth.append((bench_name, sample_dir, idx))

    real_picks = rng.sample(all_real, min(num_debug, len(all_real)))
    synth_picks = rng.sample(all_synth, min(num_debug, len(all_synth)))

    for bench_name, sample_dir in real_picks:
        result = process_real_sample(inferencer, bench_name, sample_dir, debug=True)
        if result:
            print_debug_result("REAL", bench_name, sample_dir, None, result)

    for bench_name, sample_dir, idx in synth_picks:
        result = process_synthetic_sample(inferencer, bench_name, sample_dir, idx, debug=True)
        if result:
            print_debug_result("SYNTH", bench_name, sample_dir, idx, result)


def run_full(datasets, benchmarks, overwrite):
    inferencer = load_model()
    for base in datasets:
        base = Path(base)
        n_real, n_synth = 0, 0
        for bench_name, sample_dir in discover_samples(base, benchmarks):
            r = process_real_sample(inferencer, bench_name, sample_dir, overwrite=overwrite)
            if r is not None:
                n_real += 1
            for idx in list_synthetic_indices(sample_dir):
                s = process_synthetic_sample(inferencer, bench_name, sample_dir, idx, overwrite=overwrite)
                if s is not None:
                    n_synth += 1
            if (n_real + n_synth) % 50 == 0:
                print(f"[{base.name}] real={n_real} synth={n_synth}", flush=True)
        print(f"DONE [{base.name}] real={n_real} synth={n_synth}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--mode", choices=["debug", "full"], default="debug")
    ap.add_argument("--num-debug", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "debug":
        run_debug(args.datasets, args.num_debug, args.seed)
    else:
        run_full(args.datasets, args.benchmarks, args.overwrite)


if __name__ == "__main__":
    main()
