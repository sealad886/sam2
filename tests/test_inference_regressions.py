# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


@pytest.fixture(scope="module")
def image_model():
    torch.manual_seed(20)
    return build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        device="cpu",
        apply_postprocessing=False,
    )


@pytest.fixture(scope="module")
def image_predictor(image_model):
    predictor = SAM2ImagePredictor(image_model)
    predictor.set_image(np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3))
    return predictor


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("multimask", [False, True])
def test_mask_only_prompt_batches_match_individual_predictions(
    image_predictor, batch_size, multimask
):
    grid = np.linspace(-2.0, 2.0, 256 * 256, dtype=np.float32).reshape(1, 256, 256)
    prompts = np.stack([grid + 0.25 * index for index in range(batch_size)])
    individual = [
        image_predictor.predict(
            mask_input=prompt, multimask_output=multimask, return_logits=True
        )
        for prompt in prompts
    ]
    actual = image_predictor.predict(
        mask_input=prompts, multimask_output=multimask, return_logits=True
    )
    for field, output in enumerate(actual):
        expected = np.stack([result[field] for result in individual])
        if batch_size == 1:
            expected = expected[0]
        assert output.shape == expected.shape
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=2e-6)


def test_multiple_images_accept_independent_mask_prompt_batches(image_model):
    image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    predictor = SAM2ImagePredictor(image_model)
    predictor.set_image_batch([image, image[:, ::-1].copy()])
    prompts = [np.zeros((2, 1, 256, 256)), np.ones((3, 1, 256, 256))]
    actual = predictor.predict_batch(
        mask_input_batch=prompts, multimask_output=False, return_logits=True
    )
    assert [result.shape for result in actual[0]] == [(2, 1, 16, 20), (3, 1, 16, 20)]
    assert [result.shape for result in actual[1]] == [(2, 1), (3, 1)]
    assert [result.shape for result in actual[2]] == [
        (2, 1, 256, 256),
        (3, 1, 256, 256),
    ]
    assert all(np.isfinite(result).all() for field in actual for result in field)


@pytest.mark.parametrize("crop_layers", [0, 1])
def test_automatic_masks_can_filter_every_detection(image_model, crop_layers):
    generator = SAM2AutomaticMaskGenerator(
        image_model,
        points_per_side=1,
        points_per_batch=1,
        crop_n_layers=crop_layers,
        pred_iou_thresh=float("inf"),
    )
    assert generator.generate(np.zeros((16, 20, 3), dtype=np.uint8)) == []


@pytest.fixture(scope="module")
def tracked_video(tmp_path_factory):
    torch.manual_seed(21)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        device="cpu",
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.image_size=64"],
    )
    directory = tmp_path_factory.mktemp("tracked-video")
    for index in range(24):
        Image.new("RGB", (8, 6), (20 + index, 40, 60)).save(
            directory / f"{index:05d}.jpg"
        )
    state = predictor.init_state(str(directory), offload_video_to_cpu=True)
    for object_id in (101, 205):
        predictor.add_new_mask(state, 0, object_id, torch.ones(6, 8))
    list(predictor.propagate_in_video(state))
    return predictor, state


@pytest.mark.parametrize("clear_memory", [False, True])
@pytest.mark.parametrize("corrections_as_cond", [False, True])
def test_video_corrections_clear_only_stale_memory_for_the_corrected_object(
    tracked_video, clear_memory, corrections_as_cond
):
    predictor, initial_state = tracked_video
    state = copy.deepcopy(initial_state)
    predictor.clear_non_cond_mem_around_input = clear_memory
    predictor.add_all_frames_to_correct_as_cond = corrections_as_cond
    other_frames = state["output_dict_per_obj"][1]["non_cond_frame_outputs"].copy()
    for frame_index in (10, 11):
        predictor.add_new_mask(state, frame_index, 101, torch.ones(6, 8))

    predictor.propagate_in_video_preflight(state)

    target = state["output_dict_per_obj"][0]
    expected = set(range(1, 24))
    if clear_memory:
        expected -= set(range(3, 19))
    if corrections_as_cond:
        expected -= {10, 11}
        assert set(target["cond_frame_outputs"]) == {0, 10, 11}
    else:
        expected |= {10, 11}
        assert set(target["cond_frame_outputs"]) == {0}
    assert set(target["non_cond_frame_outputs"]) == expected
    other = state["output_dict_per_obj"][1]["non_cond_frame_outputs"]
    assert other.keys() == other_frames.keys()
    assert all(other[index] is output for index, output in other_frames.items())
    assert state["obj_ids"] == [101, 205]


def test_video_propagation_can_clear_memory_around_conditioning_frames(tracked_video):
    predictor, initial_state = tracked_video
    state = copy.deepcopy(initial_state)
    predictor.clear_non_cond_mem_around_input = True
    results = list(
        predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0)
    )
    assert len(results) == 1
    frame, objects, masks = results[0]
    assert (frame, objects) == (0, [101, 205])
    assert masks.shape == (2, 1, 6, 8)
    assert torch.isfinite(masks).all()
    for outputs in state["output_dict_per_obj"].values():
        assert set(outputs["non_cond_frame_outputs"]) == set(range(8, 24))
