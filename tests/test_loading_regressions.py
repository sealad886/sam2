# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sam2.build_sam as builders
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames
from training.dataset.vos_raw_dataset import JSONRawDataset, PNGRawDataset
from training.dataset.vos_segment_loader import (
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
)


class VideoPath:
    def __init__(self, path, as_bytes=False):
        self.path = os.fsencode(path) if as_bytes else os.fspath(path)

    def __fspath__(self):
        return self.path


@pytest.mark.parametrize("source_kind", ["string", "path", "custom", "bytes_path"])
@pytest.mark.parametrize("async_loading", [False, True])
def test_jpeg_folder_pathlike_inputs_preserve_decoded_frames(
    tmp_path, source_kind, async_loading
):
    directory = tmp_path / "frames"
    directory.mkdir()
    Image.new("RGB", (8, 6), (64, 128, 192)).save(directory / "00000.jpg")
    metadata = directory / ".DS_Store"
    metadata.write_bytes(b"metadata")
    sources = {
        "string": str(directory),
        "path": directory,
        "custom": VideoPath(directory),
        "bytes_path": VideoPath(directory, as_bytes=True),
    }

    frames, height, width = load_video_frames(
        sources[source_kind],
        image_size=8,
        offload_video_to_cpu=True,
        img_mean=(0, 0, 0),
        img_std=(1, 1, 1),
        async_loading_frames=async_loading,
        compute_device="cpu",
    )

    assert (height, width) == (6, 8)
    assert len(frames) == 1
    assert frames[0].shape == (3, 8, 8)
    torch.testing.assert_close(
        frames[0].mean(dim=(1, 2)),
        torch.tensor([64, 128, 192]) / 255,
        atol=2 / 255,
        rtol=0,
    )
    assert metadata.read_bytes() == b"metadata"


@pytest.mark.parametrize("source_kind", ["path", "custom", "bytes_path"])
def test_mp4_pathlike_inputs_use_the_video_file_decoder(tmp_path, source_kind):
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height = 8, 6
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.full((6, 8, 3), (64, 128, 192), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sources = {
        "path": path,
        "custom": VideoPath(path),
        "bytes_path": VideoPath(path, as_bytes=True),
    }
    options = {"image_size": 8, "offload_video_to_cpu": True, "compute_device": "cpu"}
    expected, expected_height, expected_width = load_video_frames(str(path), **options)
    frames, height, width = load_video_frames(sources[source_kind], **options)
    assert (height, width) == (expected_height, expected_width) == (6, 8)
    torch.testing.assert_close(frames, expected)


@pytest.mark.parametrize("source_kind", ["path", "custom", "invalid"])
def test_unsupported_video_inputs_remain_rejected(tmp_path, source_kind):
    path = tmp_path / "missing"
    sources = {"path": path, "custom": VideoPath(path), "invalid": object()}
    with pytest.raises(NotImplementedError):
        load_video_frames(sources[source_kind], 8, True, compute_device="cpu")


class DownloadedModel(torch.nn.Module):
    image_size = 1024

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


@pytest.mark.parametrize(
    "kind", ["image", "video", "image_predictor", "video_predictor", "automatic_masks"]
)
@pytest.mark.parametrize("cache_kind", ["default", "string", "path"])
def test_hugging_face_entrypoints_load_weights_from_the_selected_cache(
    monkeypatch, tmp_path, kind, cache_kind
):
    selected_cache = tmp_path / "selected-cache"
    default_cache = tmp_path / "default-cache"
    observed_cache = []

    def download(repo_id, filename, *, cache_dir=None):
        assert repo_id == "facebook/sam2.1-hiera-tiny"
        assert filename == "sam2.1_hiera_tiny.pt"
        observed_cache.append(cache_dir)
        directory = default_cache if cache_dir is None else Path(cache_dir)
        directory.mkdir()
        checkpoint = directory / filename
        torch.save({"model": {"weight": torch.tensor([7.0])}}, checkpoint)
        return str(checkpoint)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr(
        builders, "instantiate", lambda *args, **kwargs: DownloadedModel()
    )
    entrypoints = {
        "image": builders.build_sam2_hf,
        "video": builders.build_sam2_video_predictor_hf,
        "image_predictor": SAM2ImagePredictor.from_pretrained,
        "video_predictor": SAM2VideoPredictor.from_pretrained,
        "automatic_masks": SAM2AutomaticMaskGenerator.from_pretrained,
    }
    options = {"device": "cpu", "apply_postprocessing": False}
    cache_dir = None
    if cache_kind != "default":
        cache_dir = str(selected_cache) if cache_kind == "string" else selected_cache
        options["cache_dir"] = cache_dir
    result = entrypoints[kind]("facebook/sam2.1-hiera-tiny", **options)
    if kind == "automatic_masks":
        model = result.predictor.model
    elif kind == "image_predictor":
        model = result.model
    else:
        model = result
    torch.testing.assert_close(model.weight, torch.tensor([7.0]))
    assert observed_cache == [cache_dir]
    assert (default_cache if cache_dir is None else selected_cache).is_dir()
    assert not (selected_cache if cache_dir is None else default_cache).exists()


def _save_mask(path, object_id=3):
    Image.fromarray(np.array([[0, object_id], [object_id, 0]], dtype=np.uint8)).save(
        path
    )


def test_palette_annotations_only_load_numeric_png_files(tmp_path):
    _save_mask(tmp_path / "00000003.png")
    _save_mask(tmp_path / "12.PNG", object_id=7)
    (tmp_path / ".DS_Store").write_bytes(b"metadata")
    (tmp_path / "notes.png").write_text("not a frame")
    (tmp_path / "9.txt").write_text("not a PNG")
    (tmp_path / "10.png").mkdir()

    loader = PalettisedPNGSegmentLoader(tmp_path)
    assert set(loader.load(3)) == {3}
    assert set(loader.load(12)) == {7}
    for ignored_id in (9, 10):
        with pytest.raises(KeyError):
            loader.load(ignored_id)
    assert (tmp_path / ".DS_Store").read_bytes() == b"metadata"


def test_png_dataset_discovers_video_directories_and_real_frames_only(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    (images / "video").mkdir(parents=True)
    (masks / "video").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "video" / "00000.jpg")
    _save_mask(masks / "video" / "00000.png")
    for directory in (images, images / "video", masks / "video"):
        (directory / ".DS_Store").write_bytes(b"metadata")
    (images / "readme.txt").write_text("not a video")
    (images / "video" / "notes.jpg").write_text("not a frame")
    (images / "video" / "00001.jpg").mkdir()

    dataset = PNGRawDataset(images, masks, frames_sampling_mult=True)
    assert len(dataset) == 1
    video, loader = dataset.get_video(0)
    assert video.video_name == "video"
    assert [frame.frame_idx for frame in video.frames] == [0]
    assert set(loader.load(0)) == {3}


def test_explicit_png_video_list_preserves_missing_video_errors(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    video_list = tmp_path / "videos.txt"
    video_list.write_text("missing\n")
    dataset = PNGRawDataset(images, masks, file_list_txt=video_list)
    assert len(dataset) == 1
    with pytest.raises(FileNotFoundError):
        dataset.get_video(0)


@pytest.mark.parametrize("dataset_type", [PNGRawDataset, JSONRawDataset])
def test_explicit_video_list_ignores_finder_metadata_but_keeps_missing_videos(
    tmp_path, dataset_type
):
    images, masks = tmp_path / "images", tmp_path / "masks"
    (images / "video").mkdir(parents=True)
    (masks / "video").mkdir(parents=True)
    metadata = images / ".DS_Store"
    metadata.write_bytes(b"metadata")
    Image.new("RGB", (2, 2)).save(images / "video" / "00000.jpg")
    _save_mask(masks / "video" / "00000.png")
    (masks / "video_manual.json").write_text(json.dumps([[]]))
    video_list = tmp_path / "videos.txt"
    video_list.write_text("video\n.DS_Store\nmissing\n")
    options = {"rm_unannotated": False} if dataset_type is JSONRawDataset else {}

    dataset = dataset_type(images, masks, file_list_txt=video_list, **options)

    assert len(dataset) == 2
    videos = {}
    missing_errors = 0
    for index in range(len(dataset)):
        try:
            video, _ = dataset.get_video(index)
        except FileNotFoundError:
            missing_errors += 1
        else:
            videos[video.video_name] = [frame.frame_idx for frame in video.frames]
    assert videos == {"video": [0]}
    assert missing_errors == 1
    assert metadata.read_bytes() == b"metadata"


@pytest.mark.parametrize("single_object", [False, True])
@pytest.mark.parametrize("sampling_multiplier", [False, True])
def test_png_object_discovery_ignores_metadata_and_non_object_entries(
    tmp_path, single_object, sampling_multiplier
):
    images, masks = tmp_path / "images", tmp_path / "masks"
    (images / "video").mkdir(parents=True)
    (masks / "video" / "0").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(images / "video" / "00000.jpg")
    _save_mask(masks / "video" / "0" / "00000.png", object_id=1)
    (masks / "video" / ".DS_Store").write_bytes(b"metadata")
    (masks / "video" / "notes.txt").write_text("not an object")
    (masks / "video" / "notes").mkdir()

    dataset = PNGRawDataset(
        images,
        masks,
        is_palette=False,
        single_object_mode=single_object,
        frames_sampling_mult=sampling_multiplier,
    )
    assert len(dataset) == 1
    video, loader = dataset.get_video(0)
    assert [frame.frame_idx for frame in video.frames] == [0]
    assert isinstance(loader, MultiplePNGSegmentLoader)
    assert set(loader.load(0)) == {1}


def test_json_dataset_discovers_video_directories_and_numeric_jpeg_frames(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    (images / "video").mkdir(parents=True)
    masks.mkdir()
    for index in (0, 2):
        Image.new("RGB", (2, 2)).save(images / "video" / f"{index:05d}.jpg")
    for directory in (images, images / "video"):
        (directory / ".DS_Store").write_bytes(b"metadata")
    (images / "video" / "1.txt").write_text("not a frame")
    (images / "video" / "notes.jpg").write_text("not a frame")
    (images / "video" / "00003.jpg").mkdir()
    (masks / "video_manual.json").write_text(json.dumps([[], [], []]))

    dataset = JSONRawDataset(images, masks, rm_unannotated=False)
    assert len(dataset) == 1
    video, _ = dataset.get_video(0)
    assert [frame.frame_idx for frame in video.frames] == [0, 2]
    assert all(Path(frame.image_path).is_file() for frame in video.frames)
