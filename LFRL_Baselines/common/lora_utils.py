from __future__ import annotations

from contextlib import contextmanager

from peft.tuners.tuners_utils import BaseTunerLayer


@contextmanager
def disabled_adapter(model):
    """Temporarily disables all LoRA adapter layers so a forward pass uses
    only the frozen base weights -- the reference-policy log-probs DPO needs.

    This project's LoRA injection uses peft.inject_adapter_in_model directly
    on model.language_model (no PeftModel wrapper, see runtime.py), so the
    high-level `PeftModel.disable_adapter()` context manager isn't available.
    Toggling BaseTunerLayer.enable_adapters directly is peft's own underlying
    mechanism and works regardless of wrapper presence.

    Since LoRA's B matrix is zero-initialized, this reference model is
    mathematically identical to the state of the model when this training run
    began -- exactly the fixed-reference-policy DPO requires -- with no need
    to keep a second full model copy in memory."""
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield
    finally:
        for layer in layers:
            layer.enable_adapters(True)
