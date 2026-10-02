from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from bagel.data.data_utils import add_special_tokens
from bagel.data.transforms import ImageTransform
from bagel.inferencer import InterleaveInferencer
from bagel.modeling.autoencoder import load_ae
from bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from bagel.modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

# Matches Bagel/train/pretrain_unified_navit.py's use_lora path exactly (LLM
# decoder only; mlp_moe_gen is the image-gen MoE branch, unused here since we
# never train visual_gen, but it reuses the same leaf names so it must be
# excluded from target_modules matching).
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_LORA_EXCLUDE_MODULES = r".*\.mlp_moe_gen\..*"


class _DeviceImageTransform:
    """ImageTransform (torchvision-backed) produces its tensor via a
    from_numpy-style path, which always lands on CPU regardless of
    torch.set_default_device -- unlike plain torch.tensor(...) calls, that
    path doesn't go through torch's device-factory dispatch. BAGEL's own
    prepare_vit_images/prepare_vae_images never move it to GPU themselves
    (the existing eval path only works because accelerate's dispatch hooks
    paper over this). This wrapper moves the transform's tensor output to the
    model's device immediately, and otherwise delegates to the original
    transform (e.g. its resize_transform helper, which returns a PIL image,
    not a tensor, and must not be touched)."""

    def __init__(self, transform, device: torch.device):
        self._transform = transform
        self._device = device

    def __call__(self, *args, **kwargs):
        out = self._transform(*args, **kwargs)
        return out.to(self._device) if torch.is_tensor(out) else out

    def __getattr__(self, name):
        return getattr(self._transform, name)


@dataclass
class BagelRuntime:
    model: Bagel
    vae_model: torch.nn.Module
    tokenizer: Qwen2Tokenizer
    new_token_ids: Dict[str, int]
    vae_transform: object
    vit_transform: object
    inferencer: InterleaveInferencer
    device: torch.device
    lora_config: Dict = field(default_factory=dict)


def load_bagel_runtime(
    *,
    model_path: str,
    device: str = "cuda:0",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    lora_exclude_modules: Optional[str] = DEFAULT_LORA_EXCLUDE_MODULES,
    lora_checkpoint_path: Optional[str] = None,
) -> BagelRuntime:
    """Loads BAGEL-7B for understanding-only (text-answer) LoRA training/rollout
    on a single GPU, following the exact model-construction and LoRA-injection
    convention already validated in this project (Bagel/train/pretrain_unified_navit.py
    and lmms_eval/models/simple/bagel.py), so checkpoints saved from here are
    directly loadable by the existing eval path with no changes there."""
    device_t = torch.device(device)

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        # visual_gen is unused (we never finetune image generation) but the
        # image-gen submodules must still exist so the base checkpoint's state
        # dict (which contains their weights) loads cleanly; we simply never
        # feed the model any VAE-path batch keys.
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

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    weights_path = os.path.join(model_path, "ema.safetensors")
    if not os.path.isfile(weights_path):
        weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    # nn.Module defaults to fp32, and load_state_dict preserves the
    # *destination* dtype (an in-place copy_ upcasts the bf16 checkpoint
    # values into fp32 params) -- so without this cast the model would sit at
    # ~2x its intended memory footprint. Cast before LoRA injection so the
    # newly created lora_A/lora_B weights are created in bf16 too, matching
    # the base layers they wrap.
    model = model.to(torch.bfloat16)

    lora_cfg_dict: Dict = {}
    if lora_rank > 0:
        from peft import LoraConfig, inject_adapter_in_model

        target_modules = lora_target_modules or DEFAULT_LORA_TARGET_MODULES
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            exclude_modules=lora_exclude_modules,
            bias="none",
        )
        # inject_adapter_in_model mutates language_model in place (same object,
        # no PeftModel wrapper) -- NOT get_peft_model, which would add a
        # "base_model.model." indirection that breaks Bagel's forward(), since
        # it reaches into language_model internals directly (e.g.
        # self.language_model.model.embed_tokens, .model.layers[i]).
        model.language_model = inject_adapter_in_model(lora_config, model.language_model, adapter_name="default")
        lora_cfg_dict = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "exclude_modules": lora_exclude_modules,
        }

        if lora_checkpoint_path:
            adapter_state_dict = load_file(os.path.join(lora_checkpoint_path, "ema.safetensors"), device="cpu")
            lm_prefix = "language_model."
            adapter_state_dict = {
                k[len(lm_prefix) :]: v for k, v in adapter_state_dict.items() if k.startswith(lm_prefix)
            }
            msg = model.language_model.load_state_dict(adapter_state_dict, strict=False)
            if msg.unexpected_keys:
                raise RuntimeError(f"LoRA checkpoint has unexpected keys: {msg.unexpected_keys}")

    model = model.to(device_t).eval()
    vae_model = vae_model.to(device_t, dtype=torch.bfloat16).eval()

    # BAGEL's own prepare_*/generate_text functions create tensors via plain
    # torch.tensor(...) with no device argument (they rely on the caller to
    # have set this). The existing eval path papers over this via accelerate's
    # device-alignment hooks (load_checkpoint_and_dispatch); we load directly
    # onto one GPU for a controllable training loop, so we set this instead --
    # deliberately after the model is already loaded and moved, so the fp32
    # load/cast steps above stay off-GPU and don't spike GPU memory.
    torch.set_default_device(device_t)

    vae_transform = _DeviceImageTransform(ImageTransform(1024, 512, 16), device_t)
    vit_transform = _DeviceImageTransform(ImageTransform(980, 224, 14), device_t)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    return BagelRuntime(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        inferencer=inferencer,
        device=device_t,
        lora_config=lora_cfg_dict,
    )
