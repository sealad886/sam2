# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import sam2.build_sam as builders
import torch
from sam2.utils.device import get_device, inference_context


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
@pytest.mark.parametrize("requested", [None, "auto"])
def test_automatic_device_priority(monkeypatch, cuda, mps, expected, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_device(requested) == torch.device(expected)


@pytest.mark.parametrize(
    "requested", ["cpu", torch.device("cpu"), "cuda:1", torch.device("mps")]
)
def test_explicit_device_wins_over_automatic_selection(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device(requested) == torch.device(requested)


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_explicit_accelerator_fails_early(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available"):
        get_device(requested)


@pytest.mark.parametrize("requested", ["cpu", "mps"])
def test_non_cuda_context_does_not_enable_autocast(monkeypatch, requested):
    autocast = Mock(
        side_effect=AssertionError("non-CUDA inference must retain float32")
    )
    monkeypatch.setattr(torch, "autocast", autocast)
    with inference_context(torch.device(requested)):
        assert (torch.ones(2, dtype=torch.float32) + 1).dtype == torch.float32


def test_cuda_context_selects_bfloat16(monkeypatch):
    autocast = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch, "autocast", autocast)
    with inference_context(torch.device("cuda")):
        pass
    autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)


class InferenceModel:
    image_size = 1024

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_model(monkeypatch):
    model = InferenceModel()
    monkeypatch.setattr(builders, "instantiate", lambda *args, **kwargs: model)
    return model


@pytest.mark.parametrize(
    "builder", [builders.build_sam2, builders.build_sam2_video_predictor]
)
def test_builder_defaults_to_available_mps(monkeypatch, fake_model, builder):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    model = builder("configs/sam2.1/sam2.1_hiera_t.yaml")
    assert model.device == torch.device("mps")


@pytest.mark.parametrize(
    "builder", [builders.build_sam2, builders.build_sam2_video_predictor]
)
def test_builder_cpu_override_is_respected(monkeypatch, fake_model, builder):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = builder("configs/sam2.1/sam2.1_hiera_t.yaml", device="cpu")
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize(
    "builder", [builders.build_sam2, builders.build_sam2_video_predictor]
)
def test_non_cuda_encoder_compilation_is_rejected_before_construction(
    monkeypatch, builder
):
    instantiate = Mock(side_effect=AssertionError("constructor must not compile"))
    monkeypatch.setattr(builders, "instantiate", instantiate)
    with pytest.raises(ValueError, match="CUDA"):
        builder(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            device="cpu",
            hydra_overrides_extra=["++model.compile_image_encoder=true"],
        )
    instantiate.assert_not_called()


def test_non_cuda_optimized_video_is_rejected_before_construction(monkeypatch):
    instantiate = Mock(side_effect=AssertionError("constructor must not compile"))
    monkeypatch.setattr(builders, "instantiate", instantiate)
    with pytest.raises(ValueError, match="CUDA"):
        builders.build_sam2_video_predictor("unused", device="cpu", vos_optimized=True)
    instantiate.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("requested", [None, "cpu"])
def test_pretrained_entrypoints_share_builder_device_selection(
    monkeypatch, fake_model, kind, requested
):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        builders,
        "_hf_download",
        lambda model_id: ("configs/sam2.1/sam2.1_hiera_t.yaml", None),
    )
    cls = SAM2ImagePredictor if kind == "image" else SAM2VideoPredictor
    kwargs = {} if requested is None else {"device": requested}
    result = cls.from_pretrained("facebook/sam2.1-hiera-tiny", **kwargs)
    model = result.model if kind == "image" else result
    assert model.device == torch.device(requested or "mps")


@pytest.mark.parametrize("late_objects", [False, True])
def test_vos_cli_uses_requested_device_and_backend_context(
    monkeypatch, tmp_path, late_objects
):
    import tools.vos_inference as vos

    (tmp_path / "video").mkdir()
    argv = [
        "vos_inference.py",
        "--base_video_dir",
        str(tmp_path),
        "--input_mask_dir",
        str(tmp_path),
        "--output_mask_dir",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    if late_objects:
        argv.append("--track_object_appearing_later_in_video")
    monkeypatch.setattr(sys, "argv", argv)
    predictor = SimpleNamespace(device=torch.device("cpu"))
    build = Mock(return_value=predictor)
    monkeypatch.setattr(vos, "build_sam2_video_predictor", build)
    active_context = []

    @contextmanager
    def context(device):
        assert device == predictor.device
        active_context.append(device)
        yield
        active_context.pop()

    def predict(**kwargs):
        assert active_context == [predictor.device]
        assert kwargs["predictor"] is predictor

    regular = Mock(side_effect=predict)
    separate = Mock(side_effect=predict)
    monkeypatch.setattr(vos, "inference_context", context)
    monkeypatch.setattr(vos, "vos_inference", regular)
    monkeypatch.setattr(vos, "vos_separate_inference_per_object", separate)
    vos.main()
    assert build.call_args.kwargs["device"] == "cpu"
    (separate if late_objects else regular).assert_called_once()
    (regular if late_objects else separate).assert_not_called()
