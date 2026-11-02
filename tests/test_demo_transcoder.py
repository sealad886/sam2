# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


av = pytest.importorskip("av")
pytest.importorskip("dataclasses_json")


@pytest.fixture
def transcoder(monkeypatch, tmp_path):
    server = Path(__file__).resolve().parents[1] / "demo/backend/server"
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "demo-data"))
    configuration_spec = importlib.util.spec_from_file_location(
        "app_conf", server / "app_conf.py"
    )
    configuration = importlib.util.module_from_spec(configuration_spec)
    monkeypatch.setitem(sys.modules, "app_conf", configuration)
    configuration_spec.loader.exec_module(configuration)
    spec = importlib.util.spec_from_file_location(
        "sam2_demo_transcoder", server / "data/transcoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mp4_path(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 16, 8, "yuv420p"
        for value in (40, 80, 120):
            frame = av.VideoFrame.from_ndarray(
                np.full((8, 16, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.mark.parametrize("rotation", [0, 90, 180, -90])
def test_video_metadata_reads_dimensions_and_display_rotation(
    transcoder, mp4_path, rotation
):
    if rotation:
        # Write a display matrix into this one-track MP4's tkhd box.
        # This fixture works without an external ffmpeg executable or a newer
        # PyAV encoding API than the supported decoder requires.
        content = bytearray(mp4_path.read_bytes())
        matrix_start = content.index(b"tkhd") + 44
        cosine = round(np.cos(np.deg2rad(rotation)) * 65536)
        sine = round(np.sin(np.deg2rad(rotation)) * 65536)
        content[matrix_start : matrix_start + 36] = struct.pack(
            ">9i", cosine, -sine, 0, sine, cosine, 0, 0, 0, 1 << 30
        )
        mp4_path.write_bytes(content)

    metadata = transcoder.get_video_metadata(str(mp4_path))

    assert (metadata.width, metadata.height) == (
        (8, 16) if abs(rotation) == 90 else (16, 8)
    )
    assert metadata.fps == 5
    assert metadata.num_video_frames == 3
    assert metadata.num_video_streams == 1
    assert metadata.duration_sec == pytest.approx(0.6)
