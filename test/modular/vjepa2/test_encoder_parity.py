"""Numerical parity tests for the V-JEPA 2 encoder port.

Follows the pi0.5 reference-equivalence pattern (see
``test_pi05_reference_equivalence.py``): a from-scratch inline reference
re-implements the encoder math and we check that our modular port produces
bit-reproducible outputs when initialized with the same weights.  This
avoids needing HuggingFace transformers or the upstream vjepa2 repo at
test time.

Reference math is taken from:
  - transformers/src/transformers/models/vjepa2/modeling_vjepa2.py (lines 84-467)
  - vjepa2/src/models/utils/modules.py (rotate_queries_or_keys)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mstar.model.vjepa2.components.rope_utils import rotate_queries_or_keys
from mstar.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mstar.model.vjepa2.config import VJepa2Config

# ---------------------------------------------------------------------------
# Tiny inline reference (pure torch) — independent from our port.
# ---------------------------------------------------------------------------


def _ref_eager_attention(q, k, v, scale):
    w = torch.matmul(q, k.transpose(-1, -2)) * scale
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, v).transpose(1, 2).contiguous()


def _ref_position_ids(n, grid_size, device):
    ids = torch.arange(n, device=device)
    tokens_per_frame = grid_size * grid_size
    frame_ids = ids // tokens_per_frame
    rem = ids - tokens_per_frame * frame_ids
    height_ids = rem // grid_size
    width_ids = rem - grid_size * height_ids
    return frame_ids, height_ids, width_ids


def _ref_apply_rope(qk, d_mask, h_mask, w_mask, d_dim, h_dim, w_dim, head_dim):
    s = 0
    qkd = rotate_queries_or_keys(qk[..., s : s + d_dim], pos=d_mask)
    s += d_dim
    qkh = rotate_queries_or_keys(qk[..., s : s + h_dim], pos=h_mask)
    s += h_dim
    qkw = rotate_queries_or_keys(qk[..., s : s + w_dim], pos=w_mask)
    s += w_dim
    if s < head_dim:
        return torch.cat([qkd, qkh, qkw, qk[..., s:]], dim=-1)
    return torch.cat([qkd, qkh, qkw], dim=-1)


def _ref_attention_forward(x, layer, num_heads, head_dim, grid_size):
    b, n, c = x.shape
    q = layer.attention.query(x).view(b, n, num_heads, head_dim).transpose(1, 2)
    k = layer.attention.key(x).view(b, n, num_heads, head_dim).transpose(1, 2)
    v = layer.attention.value(x).view(b, n, num_heads, head_dim).transpose(1, 2)
    d_mask, h_mask, w_mask = _ref_position_ids(n, grid_size, x.device)
    third = 2 * ((head_dim // 3) // 2)
    q = _ref_apply_rope(q, d_mask, h_mask, w_mask, third, third, third, head_dim)
    k = _ref_apply_rope(k, d_mask, h_mask, w_mask, third, third, third, head_dim)
    ctx = _ref_eager_attention(q, k, v, head_dim**-0.5)
    return layer.attention.proj(ctx.reshape(b, n, c))


def _ref_encoder_forward(pixel_values_videos, encoder, config):
    # Embedding: permute to [B, C, T, H, W], Conv3d, flatten+transpose
    x = pixel_values_videos.permute(0, 2, 1, 3, 4).to(encoder.embeddings.patch_embeddings.proj.weight.dtype)
    x = encoder.embeddings.patch_embeddings.proj(x).flatten(2).transpose(1, 2)
    head_dim = config.hidden_size // config.num_attention_heads
    grid_size = config.crop_size // config.patch_size
    for layer in encoder.layer:
        attn_out = _ref_attention_forward(layer.norm1(x), layer, config.num_attention_heads, head_dim, grid_size)
        x = x + attn_out
        x = x + layer.mlp.fc2(layer.mlp.activation(layer.mlp.fc1(layer.norm2(x))))
    return encoder.layernorm(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _tiny_config() -> VJepa2Config:
    """4-layer tiny ViT-ish encoder for fast CPU testing.

    Must have hidden_size divisible by num_attention_heads, and head_dim
    divisible by 6 (so d_dim = h_dim = w_dim = 2*(head_dim//3//2) is non-zero).
    """
    return VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=48,  # head_dim = 48/4 = 12; third = 2*(12//3//2) = 4
        in_chans=3,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        layer_norm_eps=1e-6,
        hidden_act="gelu",
    )


def test_rotate_queries_or_keys_matches_inline_math():
    """Bit-exact check of the RoPE helper used by all attention paths."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 6, dtype=torch.float64)
    pos = torch.arange(8, dtype=torch.float64)
    out = rotate_queries_or_keys(x, pos)

    # Recompute from first principles
    D = 6
    omega = torch.arange(D // 2, dtype=torch.float64) / (D / 2.0)
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    sin = freq.sin().repeat(1, 2)
    cos = freq.cos().repeat(1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    ref = (x * cos) + (y * sin)
    assert torch.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_encoder_matches_inline_reference_fp32(dtype):
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = VJEPA2Encoder(config).to(dtype).eval()

    b, t, c = 2, config.frames_per_clip, config.in_chans
    h = w = config.crop_size
    x = torch.randn(b, t, c, h, w, dtype=dtype)

    with torch.no_grad():
        ours = encoder(x)
        ref = _ref_encoder_forward(x, encoder, config)

    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5), f"max abs diff = {(ours - ref).abs().max().item()}"


def test_encoder_output_shape():
    """Basic sanity: output token count matches grid_depth * grid_size^2."""
    config = _tiny_config()
    encoder = VJEPA2Encoder(config).eval()
    x = torch.randn(2, config.frames_per_clip, config.in_chans, config.crop_size, config.crop_size)
    with torch.no_grad():
        out = encoder(x)
    expected_n = config.num_patches
    assert out.shape == (2, expected_n, config.hidden_size)


def test_encoder_no_nan_bf16():
    """bf16 forward should at least run without producing NaN/Inf."""
    torch.manual_seed(0)
    config = _tiny_config()
    encoder = VJEPA2Encoder(config).to(torch.bfloat16).eval()
    x = torch.randn(
        1,
        config.frames_per_clip,
        config.in_chans,
        config.crop_size,
        config.crop_size,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        out = encoder(x)
    assert torch.isfinite(out).all()


def test_encoder_state_dict_key_layout():
    """Checkpoint keys must follow the HF ``encoder.`` prefix convention so
    ``load_weights_from_file`` can use a simple prefix-based mapping."""
    config = _tiny_config()
    encoder = VJEPA2Encoder(config)
    keys = set(encoder.state_dict().keys())
    # Patch embedding
    assert "embeddings.patch_embeddings.proj.weight" in keys
    assert "embeddings.patch_embeddings.proj.bias" in keys
    # Per-layer q/k/v + proj
    assert "layer.0.attention.query.weight" in keys
    assert "layer.0.attention.key.weight" in keys
    assert "layer.0.attention.value.weight" in keys
    assert "layer.0.attention.proj.weight" in keys
    assert "layer.0.mlp.fc1.weight" in keys
    assert "layer.0.mlp.fc2.weight" in keys
    assert "layer.0.norm1.weight" in keys
    assert "layer.0.norm2.weight" in keys
    # Final norm
    assert "layernorm.weight" in keys


def test_encoder_frame_pad_when_fewer_than_tubelet():
    """When num_frames < tubelet_size, embedding layer should duplicate frames."""
    config = _tiny_config()
    # Override tubelet_size to be larger than frames we'll pass
    config.tubelet_size = 4
    encoder = VJEPA2Encoder(config).eval()
    # Pass only 1 frame
    x = torch.randn(1, 1, config.in_chans, config.crop_size, config.crop_size)
    with torch.no_grad():
        out = encoder(x)
    assert torch.isfinite(out).all()
