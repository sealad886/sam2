# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from sam2.modeling.backbones.hieradet import Hiera

from sam2.modeling.sam.transformer import RoPEAttention


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("tokens", [4, 9])
def test_mps_rotary_attention_matches_cpu_with_memory_and_object_pointers(tokens):
    torch.manual_seed(42)
    reference = RoPEAttention(
        embedding_dim=32,
        num_heads=2,
        kv_in_dim=8,
        rope_k_repeat=True,
        feat_sizes=(2, 2),
    ).eval()
    model = copy.deepcopy(reference).to("mps")
    q = torch.randn(1, tokens, 32)
    k = torch.randn(1, 2 * tokens + 3, 8)
    v = torch.randn_like(k)
    with torch.inference_mode():
        expected = reference(q, k, v, num_k_exclude_rope=3)
        actual = model(q.to("mps"), k.to("mps"), v.to("mps"), num_k_exclude_rope=3)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_image_backbone_matches_cpu_without_global_operator_fallback():
    torch.manual_seed(42)
    reference = Hiera(
        embed_dim=8,
        num_heads=1,
        stages=(1, 1, 1, 1),
        window_spec=(2, 2, 2, 2),
        global_att_blocks=(),
        window_pos_embed_bkg_spatial_size=(2, 2),
    ).eval()
    with torch.no_grad():
        reference.pos_embed.normal_()
        reference.pos_embed_window.normal_()
    model = copy.deepcopy(reference).to("mps")
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = reference(image)
        actual = model(image.to("mps"))
    assert len(actual) == len(expected)
    for result, target in zip(actual, expected):
        assert result.device.type == "mps"
        torch.testing.assert_close(result.cpu(), target, atol=3e-4, rtol=3e-4)
