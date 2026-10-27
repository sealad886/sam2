# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from sam2 import sam2_video_predictor


@pytest.mark.parametrize("backend", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("offload", [None, False, True])
def test_video_frame_storage_default_and_override(monkeypatch, backend, offload):
    predictor = SimpleNamespace(
        device=torch.device(backend),
        image_size=8,
        _get_image_feature=lambda *args, **kwargs: None,
    )
    options = {}

    def load_frames(**kwargs):
        options.update(kwargs)
        return torch.zeros(2, 3, 8, 8), 6, 8

    monkeypatch.setattr(sam2_video_predictor, "load_video_frames", load_frames)
    arguments = {} if offload is None else {"offload_video_to_cpu": offload}
    state = sam2_video_predictor.SAM2VideoPredictor.init_state(
        predictor, "unused", **arguments
    )
    expected = backend == "mps" if offload is None else offload
    assert state["offload_video_to_cpu"] is expected
    assert options["offload_video_to_cpu"] is expected
