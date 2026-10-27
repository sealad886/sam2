# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import tools.validate_inference as validation
import torch
from PIL import Image
from tools.validate_inference import mask_iou, matched_mask_fraction


def test_mask_iou_measures_pixel_agreement():
    first = np.array([[True, True], [False, False]])
    second = np.array([[True, False], [True, False]])
    assert mask_iou(first, second) == pytest.approx(1 / 3)
    assert mask_iou(first, first) == 1
    assert mask_iou(first, ~first) == 0


def test_mask_iou_rejects_different_spatial_shapes():
    with pytest.raises(ValueError):
        mask_iou(np.ones((2, 2), dtype=bool), np.ones((4,), dtype=bool))


def test_matching_cannot_reuse_one_candidate_for_multiple_reference_masks():
    foreground = np.array([[True, False], [False, False]])
    assert matched_mask_fraction([foreground, foreground], [foreground], 0.95) == 0.5


def test_matching_ignores_output_order_and_requires_threshold():
    first = np.array([[True, False], [False, False]])
    second = ~first
    assert matched_mask_fraction([first, second], [second, first], 0.95) == 1
    assert matched_mask_fraction([first], [second], 0.95) == 0


def test_missing_masks_cannot_satisfy_qualification():
    foreground = np.ones((2, 2), dtype=bool)
    assert matched_mask_fraction([foreground], [], 0.95) == 0
    with pytest.raises(ValueError):
        matched_mask_fraction([], [], 0.95)


@pytest.mark.parametrize(
    "target,bad_backend",
    [("mps", "mps"), ("mps", "cpu"), ("cpu", "cpu"), ("cpu", None)],
)
def test_bad_middle_video_session_fails_qualification(
    monkeypatch, tmp_path, target, bad_backend
):
    import sam2.build_sam
    import sam2.utils.device

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    foreground = np.array([[True, False], [False, False]])
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (2, 2)).save(image_path)
    video_path = tmp_path / "video"
    video_path.mkdir()
    for frame in range(2):
        (video_path / f"{frame:05}.jpg").touch()
    monkeypatch.setattr(sam2.utils.device, "get_device", torch.device)
    monkeypatch.setattr(sam2.utils.device, "inference_context", lambda _: nullcontext())
    monkeypatch.setattr(
        sam2.build_sam, "build_sam2_video_predictor", lambda *a, **k: None
    )
    monkeypatch.setattr(validation, "environment", lambda _: {})
    monkeypatch.setattr(validation, "timed", lambda device, function: (function(), 0.1))
    monkeypatch.setattr(validation, "cleaned_memory", lambda _: {})
    monkeypatch.setattr(
        validation, "qualify_image", lambda *a: (foreground, [foreground], {})
    )
    calls = {"cpu": 0, "mps": 0}

    def session(predictor, path, device, **kwargs):
        calls[device.type] += 1
        masks = {
            (frame, object_id): foreground for frame in range(2) for object_id in [2, 3]
        }
        # The third call is the middle measured session, following warmup.
        if device.type == bad_backend and calls[device.type] == 3:
            masks[(1, 3)] = ~foreground
        return masks, 0.1

    monkeypatch.setattr(validation, "video_session", session)
    result = validation.validate(
        SimpleNamespace(
            device=target,
            reference_device="cpu",
            image=image_path,
            video_dir=video_path,
            frames=2,
            sessions=3,
            checkpoint=tmp_path / "checkpoint.pt",
            config="unused",
        )
    )
    assert result["gates"]["video_iou_at_least_0.98"] is (bad_backend is None)
    assert result["passed"] is (bad_backend is None)
    assert calls["cpu"] == 4
