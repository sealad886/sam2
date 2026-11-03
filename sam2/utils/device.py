# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch


def get_device(device=None):
    """Resolve an explicit device or automatically prefer CUDA, MPS, then CPU."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and check "
            "your NVIDIA driver, or select device='cpu' or device='auto'."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Use an MPS-enabled PyTorch build on a supported "
            "Mac, or select device='cpu' or device='auto'."
        )
    return device


def inference_context(device):
    """Use CUDA bfloat16 autocast; retain float32 inference on MPS and CPU."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
