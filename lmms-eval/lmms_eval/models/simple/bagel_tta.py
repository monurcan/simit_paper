"""Bagel + single-sample test-time adaptation (TTA): for each eval doc, the
same retrieved/synthesized (image, question, answer) examples that
bagel_icl.py would show as in-context demonstrations are instead used to
LoRA-finetune the model for that one doc only. The model answers the doc's
real query (zero-shot prompt, no few-shot demos -- identical prompting to the
pretrained baseline), then the LoRA adapter is reset to its initial
(zero-effect) state before the next doc, so every doc starts from the exact
same pretrained weights.

This deliberately reuses bagel_icl.py's retrieval machinery unchanged
(ICLRetriever / SyntheticSampleRetriever / DFASCSyntheticSampleRetriever /
RealSampleRetriever, and all of BagelICL's __init__ argument handling for
picking one of them) and Bagel's own production training building blocks
(bagel.data.vlm_dataset.SftJSONLIterableDataset, bagel.data.dataset_base.
PackedDataset, the same vlm_sft JSONL convention as Bagel/prepare_vqa_data.py)
rather than reimplementing tokenization/packing. Only the LoRA-adapter
lifecycle (inject once, train per-doc, snapshot-reset per-doc) and the
per-doc training loop (plain single-GPU AdamW, no FSDP/cpu_offload -- unlike
Bagel/train/pretrain_unified_navit.py, this runs on the same already-loaded
inference model/process, so no distributed setup is needed or wanted) are
new.

    accelerate launch -m lmms_eval \\
        --model bagel_tta \\
        --model_args pretrained=/path/to/BAGEL-7B-MoT,icl_samples_root=$SIMIT_DATA/generated_samples_discard_unverified_v2,num_icl_examples=4,num_generations=6,adapt_steps=20,adapt_lr=2e-4,lora_rank=8,lora_alpha=16 \\
        --tasks ai2d_lite \\
        --batch_size 1 \\
        --output_path ./logs/

Only "understanding" and "think_understanding" modes are supported, same
restriction as bagel_icl.py.
"""
import gc
import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.bagel import BAGEL_AVAILABLE
from lmms_eval.models.simple.bagel_icl import BagelICL, ICLExample

if BAGEL_AVAILABLE:
    from bagel.data.data_utils import get_flattened_position_ids_extrapolate, pil_img2rgb
    from bagel.data.dataset_base import DataConfig, PackedDataset, SimpleCustomBatch
    from bagel.data.transforms import ImageTransform
    from bagel.data.vlm_dataset import SftJSONLIterableDataset
    from peft import LoraConfig, inject_adapter_in_model
    from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_with_min_lr_schedule_with_warmup


@register_model("bagel_tta")
class BagelTTA(BagelICL):
    """Bagel VQA evaluation with single-sample test-time LoRA adaptation.

    Subclasses BagelICL purely to reuse its retrieval setup: every
    __init__ kwarg BagelICL understands (icl_samples_root, num_icl_examples,
    retrieval_strategy, num_generations, df_enabled/asc_enabled and their
    epsilon/A0/A1/B/T_low/T_high, rag_index_root/rag_embed_python/...) works
    identically here and picks the same retriever -- "num_icl_examples" is
    kept as the argument name (unchanged from BagelICL) but here it means
    "number of adaptation/finetuning examples", not in-context demo count.

    generate_until does NOT call BagelICL's ICL-prompt-building path at all:
    the query prompt is exactly BagelUMM's plain baseline prompt (no demos).
    Instead, per doc: reset the LoRA adapter to its pristine (no-op) state,
    LoRA-finetune on the retrieved examples for adapt_steps steps, generate
    the answer, and leave the adapter in place until the next *distinct* doc
    (repeated rollouts of the same doc, e.g. self-consistency sampling, reuse
    the same adapted weights rather than re-training per rollout).
    """

    def __init__(
        self,
        # LoRA adapter hyperparameters (see Bagel/train/pretrain_unified_navit.py's
        # --use_lora training path; defaults mirror its own field defaults except
        # lora_dropout, which is 0.0 here to match the winning production configs
        # summarized in the LoRA-vs-full-finetune artifact -- every winning DF+ASC
        # row there used dropout=0.0).
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules_csv: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        lora_exclude_modules_regex: str = r".*\.mlp_moe_gen\..*",
        # Per-doc adaptation training loop. These have no established "best"
        # values yet (this is a new setting -- production sweeps so far only
        # cover full-dataset LoRA finetuning and frozen-model SG-ICL); the
        # defaults below are reasonable starting points for sweeping, not
        # tuned results.
        adapt_steps: int = 20,
        adapt_lr: float = 2e-4,
        adapt_lr_scheduler: str = "constant",  # "constant" | "cosine"
        adapt_warmup_steps: int = 0,
        adapt_min_lr: float = 1e-7,
        adapt_beta1: float = 0.9,
        adapt_beta2: float = 0.95,
        adapt_eps: float = 1e-15,
        adapt_max_grad_norm: float = 1.0,
        # Adaptation-image transform args -- match Bagel/data/configs/vqa_*_v2_dfadaptive_*.yaml's
        # image_transform_args (the production LoRA-finetuning data pipeline), which differs from
        # the *query* image's own transform (unaffected -- still InterleaveInferencer's usual
        # vit_transform, untouched by this class). min_image_size lowered from that yaml's 378 to
        # 224 (matching the query transform's own floor, ImageTransform(980, 224, 14) in bagel.py)
        # so icl_image_max_long_side below actually sticks: MaxLongEdgeMinShortEdgeResize can
        # *upscale* a pre-shrunk image back up if its short edge falls below min_image_size, which
        # would partially undo a small icl_image_max_long_side cap for images with an aspect ratio
        # much wider than square (e.g. min_image_size=378 fights any cap under ~672 for 16:9 images).
        adapt_image_max_size: int = 980,
        adapt_image_min_size: int = 224,
        adapt_image_stride: int = 14,
        adapt_image_max_pixels: int = 2_007_040,
        # Overrides BagelICL's own default of None (unlimited): that default is fine for BagelICL's
        # frozen-inference demo images (shown once per query via KV-cache, no backward pass), but
        # this class packs up to num_icl_examples adaptation images into ONE sequence and runs full
        # forward+backward through the whole model on it -- activation memory for that scales with
        # total packed sequence length, so training needs a real per-image token cap by default.
        # Confirmed necessary in production: full-resolution adaptation images caused a CUDA OOM on
        # chartqa/infovqa docs (see run_bagel_tta_eval.sh's PYTORCH_CUDA_ALLOC_CONF note) even after
        # that allocator-level mitigation; capping the images at the source is the more direct fix.
        icl_image_max_long_side: int = 512,
        adapt_scratch_dir: str = "",
        **kwargs,
    ) -> None:
        if not BAGEL_AVAILABLE:
            raise ImportError("bagel_tta requires the same Bagel dependencies as bagel_umm/bagel_icl (see their import error above).")

        # Set before super().__init__() -- BagelUMM.__init__ (reached via
        # BagelICL's own super().__init__ call) invokes self._load_model(),
        # which our override below reads these from.
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = [m.strip() for m in lora_target_modules_csv.split(",") if m.strip()]
        self.lora_exclude_modules = lora_exclude_modules_regex or None

        self.adapt_steps = int(adapt_steps)
        self.adapt_lr = float(adapt_lr)
        self.adapt_lr_scheduler = adapt_lr_scheduler
        self.adapt_warmup_steps = int(adapt_warmup_steps)
        self.adapt_min_lr = float(adapt_min_lr)
        self.adapt_beta1 = float(adapt_beta1)
        self.adapt_beta2 = float(adapt_beta2)
        self.adapt_eps = float(adapt_eps)
        self.adapt_max_grad_norm = float(adapt_max_grad_norm)

        self._adapt_image_transform_args = dict(
            max_image_size=int(adapt_image_max_size),
            min_image_size=int(adapt_image_min_size),
            image_stride=int(adapt_image_stride),
            max_pixels=int(adapt_image_max_pixels),
        )

        self._adapted_key = None
        self._last_adapt_info: Dict[str, Any] = {}
        self._scratch_root = adapt_scratch_dir or tempfile.mkdtemp(prefix="bagel_tta_scratch_")
        os.makedirs(self._scratch_root, exist_ok=True)

        super().__init__(icl_image_max_long_side=icl_image_max_long_side, **kwargs)

        self._adapt_vit_transform = ImageTransform(**self._adapt_image_transform_args)

        # Pristine (zero-effect) LoRA state: lora_B is zero-initialized by
        # peft, so at this point the adapter contributes nothing and the
        # model is numerically identical to the unmodified pretrained model
        # (lora_A's random init never matters while lora_B stays at zero).
        # Reloading this exact snapshot before every doc is what makes the
        # "reset" semantics exact, rather than relying on re-running peft's
        # own (RNG-based) init routine.
        self._lora_init_state = {k: v.detach().clone() for k, v in self.model.language_model.state_dict().items() if "lora_" in k}

        eval_logger.info(
            f"BagelTTA initialized: {len(self._lora_init_state)} LoRA tensors snapshotted for reset; "
            f"lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}, "
            f"adapt_steps={self.adapt_steps}, adapt_lr={self.adapt_lr}, adapt_lr_scheduler={self.adapt_lr_scheduler}, "
            f"num_adapt_examples={self.num_icl_examples}, scratch_root={self._scratch_root}"
        )

    def _load_model(self):
        super()._load_model()  # BagelUMM._load_model: loads base weights, builds self.inferencer

        model = self.model

        # Freeze *everything* first, not just vit_model/connector (as
        # Bagel/train/pretrain_unified_navit.py's --freeze_vit/--freeze_connector
        # do): that script never explicitly freezes model.vit_pos_embed (a
        # PositionEmbedding sibling of vit_model/connector, not nested inside
        # either), relying on it apparently not mattering there. For this
        # class it must be frozen -- the whole point of the reset-per-doc
        # design is that nothing outside the LoRA adapter can ever drift.
        for p in model.parameters():
            p.requires_grad = False
        model.vit_model.eval()
        model.connector.eval()

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            exclude_modules=self.lora_exclude_modules,
            bias="none",
        )
        # inject_adapter_in_model mutates language_model in place (same
        # object, no PeftModel wrapper) -- matches bagel.py::_apply_lora_adapter
        # and pretrain_unified_navit.py's own training-time injection, and is
        # required for Bagel.forward's hardcoded attribute paths (e.g.
        # self.language_model.model.embed_tokens) to keep working.
        model.language_model = inject_adapter_in_model(lora_config, model.language_model, adapter_name="default")

        trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.language_model.parameters())
        eval_logger.info(f"BagelTTA: injected trainable LoRA adapter into language_model, " f"{trainable:,}/{total:,} params trainable ({100 * trainable / max(total, 1):.4f}%)")

    def _reset_lora(self) -> None:
        self.model.language_model.load_state_dict(self._lora_init_state, strict=False)

    def _build_adapt_jsonl(self, examples: List[ICLExample], scratch_dir: Path) -> Path:
        """Writes examples to the same vlm_sft JSONL convention as
        Bagel/prepare_vqa_data.py (relative-or-absolute image path(s) +
        human/gpt conversation turns), so SftJSONLIterableDataset (unmodified
        production code) can read them directly.

        Always re-saves through self._resize_demo_image (inherited from
        BagelICL, downscale-only/aspect-preserving/no-op above the cap) rather
        than referencing SyntheticSampleRetriever's image_path in place: these
        images get packed together into one training sequence and backpropped
        through the whole model, unlike BagelICL's own use of this same method
        for frozen-inference demo images, so their token cost matters much more
        here (see icl_image_max_long_side's docstring in __init__)."""
        jsonl_path = scratch_dir / "adapt.jsonl"
        with jsonl_path.open("w") as f:
            for i, ex in enumerate(examples):
                raw_imgs = ex.images if ex.images is not None else [pil_img2rgb(Image.open(ex.image_path))]
                img_paths = []
                for j, img in enumerate(raw_imgs):
                    resized = self._resize_demo_image(pil_img2rgb(img))
                    p = scratch_dir / f"ex{i}_img{j}.png"
                    resized.save(p)
                    img_paths.append(str(p))
                image_field = img_paths if len(img_paths) > 1 else img_paths[0]
                value = "<image>" * len(img_paths) + f"\n{ex.question}"
                record = {
                    "image": image_field,
                    "conversations": [
                        {"from": "human", "value": value},
                        {"from": "gpt", "value": ex.answer},
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonl_path

    def _run_adaptation(self, examples: List[ICLExample]) -> Dict[str, Any]:
        """LoRA-finetunes self.model on `examples` for self.adapt_steps plain
        (non-FSDP, single-GPU) AdamW steps, reusing Bagel's own production
        data pipeline (SftJSONLIterableDataset + PackedDataset.pack_sequence)
        for tokenization/packing and Bagel.forward for the loss -- only the
        training *loop* itself (no distributed setup, no cpu_offload, no EMA:
        we want the live trained weights to answer with, not an EMA copy that
        would barely have moved after a handful of steps) is new code.

        Packs all of `examples` into a single fixed batch once and runs every
        step on that same batch (rather than re-invoking the streaming
        dataset per step, as the production training script does): the two
        are behaviorally identical here since SftJSONLIterableDataset's
        vit_cond_dropout_prob/text_cond_dropout_prob are inert for this
        vlm_sft format (they only ever apply when a sequence_plan item has
        enable_cfg=1, which this dataset class never sets), and doing it once
        avoids repeatedly re-reading/re-transforming the same few images.
        """
        scratch_dir = Path(tempfile.mkdtemp(prefix="doc_", dir=self._scratch_root))
        try:
            jsonl_path = self._build_adapt_jsonl(examples, scratch_dir)
            sft_dataset = SftJSONLIterableDataset(
                dataset_name="bagel_tta_adapt",
                transform=self._adapt_vit_transform,
                tokenizer=self.tokenizer,
                frame_sampler=None,
                jsonl_path_list=[str(jsonl_path)],
                data_dir_list=[""],
                num_used_data=[len(examples)],
                local_rank=0,
                world_size=1,
                num_workers=1,
                shuffle_lines=False,
                shuffle_seed=0,
                preserve_order=True,
            )
            samples = list(itertools.islice(iter(sft_dataset), len(examples)))
            if not samples:
                return {"num_examples": 0, "loss_history": []}

            # PackedDataset.__new__ bypasses __init__ (which would otherwise
            # require a DATASET_INFO/DATASET_REGISTRY-backed dataset config
            # file -- overkill for a handful of already-in-hand examples):
            # we only need its pack_sequence/set_sequence_status/to_tensor
            # instance methods, which just read the attributes set below.
            packer = PackedDataset.__new__(PackedDataset)
            packer.tokenizer = self.tokenizer
            packer.data_config = DataConfig(grouped_datasets={})
            packer.interpolate_pos = False
            packer.get_flattened_position_ids = get_flattened_position_ids_extrapolate
            packer.use_flex = False
            for k, v in self._new_token_ids.items():
                setattr(packer, k, v)

            seq_status = packer.set_sequence_status()
            for sample in samples:
                seq_status = packer.pack_sequence(sample, seq_status)
            packed = packer.to_tensor(seq_status)
            packed["batch_data_indexes"] = [s["data_indexes"] for s in samples]

            batch = SimpleCustomBatch([packed]).cuda(self.device).to_dict()
            batch.pop("batch_data_indexes", None)
            batch.pop("ce_loss_weights", None)

            if batch.get("ce_loss_indexes") is None or batch["ce_loss_indexes"].numel() == 0:
                eval_logger.warning("[BagelTTA] adaptation batch has no loss tokens (empty/degenerate examples), skipping training")
                return {"num_examples": len(samples), "loss_history": [], "sequence_length": packed["sequence_length"]}

            lm = self.model.language_model
            lm.train()
            params = [p for p in lm.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=self.adapt_lr, betas=(self.adapt_beta1, self.adapt_beta2), eps=self.adapt_eps, weight_decay=0.0)
            if self.adapt_lr_scheduler == "cosine":
                scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                    optimizer, num_warmup_steps=self.adapt_warmup_steps, num_training_steps=max(self.adapt_steps, 1), min_lr=self.adapt_min_lr
                )
            else:
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.adapt_warmup_steps)

            total_ce_tokens = batch["ce_loss_indexes"].numel()
            loss_history = []
            # BagelUMM._load_model always builds BagelConfig with visual_gen=True
            # (the eval model supports every mode, not just understanding), but our
            # packed batch never has padded_latent/patchified_vae_latent_shapes (this
            # is an understanding-only SFT batch, same as Bagel/train/pretrain_unified_navit.py's
            # own --visual_gen False runs) -- Bagel.forward's visual_gen branch would
            # crash trying to iterate those Nones. Flip it off only for this forward
            # pass; restored in `finally` below so InterleaveInferencer's own inference
            # path (which already works correctly with visual_gen=True) is untouched.
            original_visual_gen = self.model.config.visual_gen
            self.model.config.visual_gen = False
            try:
                for _ in range(self.adapt_steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=True, dtype=self._torch_dtype):
                        loss_dict = self.model(**batch)
                    ce = loss_dict["ce"].sum() / total_ce_tokens
                    ce.backward()
                    torch.nn.utils.clip_grad_norm_(params, self.adapt_max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    loss_history.append(ce.item())
            finally:
                self.model.config.visual_gen = original_visual_gen

            lm.eval()
            del optimizer, scheduler, loss_dict, ce, batch
            # Autograd graphs create Tensor<->grad_fn reference cycles that plain
            # refcounting can't free immediately; empty_cache() alone only returns
            # memory the allocator already knows is unreferenced. Cheap defensive
            # practice per doc -- NOT, on its own, what fixed the chartqa/infovqa
            # OOMs seen in production (that turned out to be genuine peak-memory
            # need from large uncapped adaptation images, fixed by the
            # icl_image_max_long_side resize in _build_adapt_jsonl instead; see
            # _run_adaptation_with_oom_retry for the remaining per-doc fallback).
            gc.collect()
            torch.cuda.empty_cache()
            return {"num_examples": len(samples), "loss_history": loss_history, "sequence_length": packed["sequence_length"]}
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    # Fallback cap sizes tried, in order, only after a CUDA OOM during a doc's
    # adaptation -- confirmed in production that OOMs here are driven by per-
    # image token count (large adaptation images packed together for training,
    # unlike frozen-inference demos which never backprop), not fragmentation:
    # an allocator-level mitigation alone did not stop an identical crash, while
    # capping image size at the source (icl_image_max_long_side) did. Applies
    # only to the one doc that actually OOM'd; every other doc in the same eval
    # run still uses the configured (default 512) cap at full speed.
    _OOM_RETRY_RESIZE_LADDER = (256, 128)

    def _adapt_for_doc(self, task: str, doc_id: Any, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Resets + retrains only when the doc actually changes, so repeated
        rollouts of the same doc (e.g. self-consistency sampling) reuse the
        same adapted weights instead of re-training per rollout."""
        key = (task, doc_id)
        if key == self._adapted_key:
            return self._last_adapt_info

        examples = self.retriever.retrieve(task, doc_id, doc, self.num_icl_examples)
        info: Dict[str, Any] = {"num_examples": len(examples), "examples": examples, "loss_history": []}
        if examples:
            info.update(self._run_adaptation_with_oom_retry(task, doc_id, examples))
        else:
            self._reset_lora()

        self._adapted_key = key
        self._last_adapt_info = info
        return info

    def _run_adaptation_with_oom_retry(self, task: str, doc_id: Any, examples: List[ICLExample]) -> Dict[str, Any]:
        """Runs _run_adaptation, and on a CUDA OOM, resets (undoing any partial
        training from the failed attempt) and retries the SAME doc with
        progressively smaller adaptation images (_OOM_RETRY_RESIZE_LADDER),
        never touching the configured default for any other doc. If even the
        smallest size still OOMs, gives up and falls back to zero-shot
        (unadapted) generation for this one doc rather than crashing the
        whole benchmark run."""
        caps = (self.icl_image_max_long_side,) + self._OOM_RETRY_RESIZE_LADDER
        original_cap = self.icl_image_max_long_side
        try:
            for attempt, cap in enumerate(caps):
                self._reset_lora()
                self.icl_image_max_long_side = cap
                try:
                    result = self._run_adaptation(examples)
                    if attempt > 0:
                        eval_logger.warning(f"[BagelTTA] task={task} doc_id={doc_id}: adaptation OOM'd at icl_image_max_long_side={caps[attempt - 1]}, " f"succeeded on retry with icl_image_max_long_side={cap}")
                    return result
                except torch.cuda.OutOfMemoryError as e:
                    eval_logger.warning(
                        f"[BagelTTA] task={task} doc_id={doc_id}: adaptation OOM'd at icl_image_max_long_side={cap} "
                        f"({len(examples)} example(s)): {e}. "
                        + ("Retrying with smaller images." if attempt + 1 < len(caps) else "Exhausted resize ladder, falling back to zero-shot for this doc.")
                    )
                    # An OOM that happens mid-backward() can leave autograd's
                    # background CUDA work queue mid-flight; PyTorch does not
                    # guarantee clean recovery from that in the same process
                    # (see e.g. https://github.com/pytorch/pytorch/issues/18853),
                    # and confirmed in production: a trial recovered from an
                    # in-process OOM retry here, kept running correctly for
                    # ~130 more docs, then hung indefinitely (high CPU, zero
                    # progress, no further exception) on a later doc -- a
                    # delayed symptom of the same failed backward, not a fresh
                    # unrelated OOM. Synchronizing here forces any leftover
                    # async work from the failed backward to fully settle
                    # before we touch the GPU again, rather than papering over
                    # it and hoping the next backward doesn't inherit it.
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
            self._reset_lora()
            # _run_adaptation always leaves lm in train() mode while its own
            # training loop runs (see lm.train()/lm.eval() above) -- when every
            # attempt OOMs, execution never reaches _run_adaptation's own
            # lm.eval() call, so the model is left training-mode-sticky across
            # into generate_until's subsequent interleave_inference call.
            # Qwen2Model.forward dispatches on self.training (forward_train
            # takes different kwargs than forward_inference), so an unadapted
            # inference request in this state hard-crashes with a TypeError
            # rather than degrading gracefully to zero-shot as intended.
            self.model.language_model.eval()
            return {"num_examples": len(examples), "loss_history": [], "oom_fallback_exhausted": True}
        finally:
            self.icl_image_max_long_side = original_cap

    def _dump_tta_prompt(
        self,
        dump_dir: str,
        task: str,
        doc_id: Any,
        adapt_info: Dict[str, Any],
        query_input_list: List[Any],
        output_text: str,
        target: Optional[str],
    ) -> None:
        sample_dir = Path(dump_dir) / task / f"doc{doc_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        parts = []
        for part in query_input_list:
            if isinstance(part, Image.Image):
                img_name = f"query_img{len(parts)}.png"
                part.save(sample_dir / img_name)
                parts.append({"type": "image", "path": img_name})
            else:
                parts.append({"type": "text", "text": part})

        adapt_examples = []
        for i, ex in enumerate(adapt_info.get("examples", [])):
            entry = {"question": ex.question, "answer": ex.answer}
            if ex.image_path:
                entry["image_path"] = ex.image_path
            elif ex.images is not None:
                img_names = []
                for j, img in enumerate(ex.images):
                    img_name = f"adapt_ex{i}_img{j}.png"
                    pil_img2rgb(img).save(sample_dir / img_name)
                    img_names.append(img_name)
                entry["images"] = img_names
            adapt_examples.append(entry)

        record = {
            "task": task,
            "doc_id": doc_id,
            "num_adapt_examples": adapt_info.get("num_examples", 0),
            "adapt_examples": adapt_examples,
            "loss_history": adapt_info.get("loss_history", []),
            "sequence_length": adapt_info.get("sequence_length"),
            "query_parts": parts,
            "model_output": output_text,
            "target": target,
        }
        with open(sample_dir / "prompt.json", "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (TTA)")
        dump_dir = os.environ.get("BAGEL_TTA_PROMPT_DUMP_DIR")

        rollout_counts: Dict[Any, int] = {}
        for request in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            rollout_key = (task, doc_id)
            rollout_idx = rollout_counts.get(rollout_key, 0)
            rollout_counts[rollout_key] = rollout_idx + 1
            force_greedy = request.repeats is not None and request.repeats > 1 and rollout_idx == 0

            doc = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(doc)

            adapt_info = self._adapt_for_doc(task, doc_id, doc)

            # Plain baseline prompt -- no ICL demos prepended (unlike
            # BagelICL.generate_until, which calls _build_icl_input_list).
            query_input_list = self._build_input_list(contexts, visuals)

            inference_params = self.inference_params.copy()
            inference_params["text_temperature"] = self.text_temperature
            if force_greedy:
                inference_params["do_sample"] = False

            eval_logger.debug(
                f"[generate_until TTA] task={task} doc_id={doc_id} "
                f"adapted_with={adapt_info.get('num_examples', 0)} example(s), "
                f"loss_history={adapt_info.get('loss_history')}, input_list={query_input_list}"
            )

            with torch.autocast(device_type="cuda", enabled=True, dtype=self._torch_dtype):
                output_list = self.inferencer.interleave_inference(query_input_list, **inference_params)

            output_text = ""
            for output in output_list:
                if isinstance(output, str):
                    output_text = output

            if dump_dir:
                self._dump_tta_prompt(dump_dir, task, doc_id, adapt_info, query_input_list, output_text, self._dump_target(task, doc))

            res.append(output_text)
            pbar.update(1)

        pbar.close()
        # Leave the model in its pristine state for any subsequent
        # generate_until call (e.g. a second --tasks entry in the same run).
        self._reset_lora()
        return res
