from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from safetensors.torch import save_file


def save_lora_checkpoint(model, save_dir: str, lora_config: Dict) -> None:
    """Saves only LoRA adapter weights, in the exact format lmms-eval's
    bagel_umm model expects for `lora_ckpt_path` (see
    lmms_eval/models/simple/bagel.py's _apply_lora_adapter): a directory with
    lora_config.json + ema.safetensors, the latter containing only
    'language_model.'-prefixed keys with 'lora_' in their name. A checkpoint
    saved here is directly evaluable with the existing eval scripts, no
    changes needed there."""
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.language_model.state_dict()
    lora_state_dict = {
        f"language_model.{k}": v.detach().cpu().contiguous() for k, v in state_dict.items() if "lora_" in k
    }
    save_file(lora_state_dict, os.path.join(save_dir, "ema.safetensors"))
    with open(os.path.join(save_dir, "lora_config.json"), "w", encoding="utf-8") as f:
        json.dump(lora_config, f, indent=2)


def save_training_state(optimizer, scheduler, step: int, save_dir: str) -> None:
    """Saves optimizer + scheduler state + step counter alongside the LoRA
    checkpoint, so a killed/resubmitted job can resume exactly instead of
    restarting from step 0. Optimizer state here only covers the LoRA
    parameters (a few hundred small tensors), not the frozen 7B base model,
    so this stays cheap in both file size and transient memory -- no full
    model state is ever gathered."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    with open(os.path.join(save_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step}, f)


def load_training_state(save_dir: str, optimizer, scheduler) -> Optional[int]:
    """Loads optimizer + scheduler state in place and returns the saved step
    count, or None if this checkpoint has no resume state (e.g. it was saved
    before this feature existed) -- callers should treat None as "start
    fresh", not an error, so old checkpoints keep working unchanged."""
    opt_path = os.path.join(save_dir, "optimizer.pt")
    sched_path = os.path.join(save_dir, "scheduler.pt")
    state_path = os.path.join(save_dir, "training_state.json")
    if not (os.path.exists(opt_path) and os.path.exists(sched_path) and os.path.exists(state_path)):
        return None
    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)["step"]
