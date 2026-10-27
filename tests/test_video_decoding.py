# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from sam2.utils.misc import load_video_frames


av = pytest.importorskip("av")


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("offload", [False, True])
def test_mp4_paths_and_bytes_preserve_frames_dimensions_and_device(
    tmp_path, as_bytes, backend, offload
):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    path = tmp_path / "clip.mp4"
    values = [(40, 80, 120), (100, 140, 180), (180, 120, 60)]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width, stream.height = 8, 6
        stream.pix_fmt = "yuv420p"
        for value in values:
            frame = av.VideoFrame.from_ndarray(
                np.full((6, 8, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = path.read_bytes() if as_bytes else str(path)
    frames, height, width = load_video_frames(
        video,
        image_size=8,
        offload_video_to_cpu=offload,
        img_mean=(0, 0, 0),
        img_std=(1, 1, 1),
        compute_device=backend,
    )
    assert (height, width) == (6, 8)
    assert frames.shape == (3, 3, 8, 8)
    assert frames.dtype == torch.float32
    assert frames.device.type == ("cpu" if offload else backend)
    for frame, value in zip(frames.cpu(), values):
        torch.testing.assert_close(
            frame.mean(dim=(1, 2)),
            torch.tensor(value) / 255,
            atol=5 / 255,
            rtol=0,
        )
