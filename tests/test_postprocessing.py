# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import sam2
import torch
from PIL import Image

from sam2.utils.misc import (
    fill_holes_in_mask_scores,
    get_connected_components,
    load_video_frames_from_jpg_images,
)
from sam2.utils.transforms import SAM2Transforms


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize("even_size", [False, True])
def test_components_preserve_connectivity_areas_batch_and_device(device, even_size):
    mask = torch.tensor(
        [
            [[[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]],
        ],
        dtype=torch.bool,
        device=device,
    )
    if even_size:
        mask = torch.nn.functional.pad(mask, (0, 1, 0, 1))
    labels, areas = get_connected_components(mask)
    assert labels.shape == areas.shape == mask.shape
    assert labels.dtype == areas.dtype == torch.int32
    assert labels.device == areas.device == mask.device
    labels, areas = labels.cpu(), areas.cpu()
    assert torch.equal(labels > 0, mask.cpu())
    assert labels[0, 0, 0, 0] == labels[0, 0, 1, 1]
    assert labels[0, 0, 0, 0] != labels[0, 0, 0, 4]
    assert torch.equal(areas[0], mask.cpu()[0].int() * 2)
    assert torch.equal(areas[1], mask.cpu()[1].int())


@pytest.mark.parametrize("shape", [(0, 1, 3, 5), (1, 1, 0, 5), (2, 1, 4, 4)])
def test_empty_components(shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    labels, areas = get_connected_components(mask)
    assert labels.shape == areas.shape == mask.shape
    assert not labels.any() and not areas.any()


def test_noncontiguous_full_components():
    mask = torch.ones((2, 1, 3, 5), dtype=torch.uint8).transpose(-1, -2)
    labels, areas = get_connected_components(mask)
    assert (labels > 0).all()
    assert (areas == 15).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_missing_cuda_extension_retains_cleanup(monkeypatch):
    import sys

    monkeypatch.delattr(sam2, "_C", raising=False)
    monkeypatch.setitem(sys.modules, "sam2._C", None)
    mask = torch.zeros((1, 1, 4, 4), device="cuda", dtype=torch.bool)
    mask[0, 0, 0, 0] = mask[0, 0, 1, 1] = True
    labels, areas = get_connected_components(mask)
    assert labels.device == areas.device == mask.device
    assert labels.dtype == areas.dtype == torch.int32
    assert torch.equal(labels > 0, mask)
    assert torch.equal(areas, mask.int() * 2)


def test_holes_and_sprinkles_are_processed_without_cuda_extension(device):
    scores = torch.full((1, 1, 5, 5), 3.0, device=device)
    scores[0, 0, 2, 2] = -3.0
    filled = fill_holes_in_mask_scores(scores, max_area=1)
    assert filled[0, 0, 2, 2] > 0
    assert scores[0, 0, 2, 2] < 0
    scores = -scores
    transform = SAM2Transforms(5, mask_threshold=0, max_sprinkle_area=1)
    result = transform.postprocess_masks(scores, (5, 5))
    assert (result < 0).all()


@pytest.mark.parametrize("async_loading", [False, True])
def test_jpeg_loading_is_float32_and_matches_cpu(tmp_path, device, async_loading):
    pixels = np.full((6, 8, 3), [64, 128, 192], dtype=np.uint8)
    for i in range(2):
        Image.fromarray(pixels).save(tmp_path / f"{i:05d}.jpg")
    frames, height, width = load_video_frames_from_jpg_images(
        str(tmp_path),
        8,
        False,
        async_loading_frames=async_loading,
        compute_device=device,
    )
    if async_loading:
        frames.thread.join(timeout=10)
        assert not frames.thread.is_alive()
    assert (height, width) == (6, 8)
    frame = frames[1]
    assert frame.dtype == torch.float32
    assert frame.device.type == device.type
    reference, _, _ = load_video_frames_from_jpg_images(
        str(tmp_path), 8, True, compute_device="cpu"
    )
    torch.testing.assert_close(frame.cpu(), reference[1])
