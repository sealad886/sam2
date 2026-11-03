# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from pathlib import Path

import pytest
import torch


def hosted_mps_first_allocation_unavailable(
    error, allocated_before, platform, github_actions, runner_environment
):
    """Recognize the hosted VM's documented failure before any GPU allocation."""
    return (
        platform == "darwin"
        and github_actions == "true"
        and runner_environment == "github-hosted"
        and allocated_before == 0
        and isinstance(error, RuntimeError)
        and str(error).startswith("MPS backend out of memory (MPS allocated: 0 bytes,")
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "mps: requires a functional Apple MPS GPU")


def _report_mps_preflight(config, status, detail):
    message = f"MPS preflight, PyTorch {torch.__version__}: {status}. {detail}"
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as output:
            output.write(f"### MPS hardware preflight\n\n{message}\n\n")
            output.write(
                "Hosted platform tests do not replace native Apple Silicon "
                "qualification. Unavailable hardware cases are reported as skips.\n\n"
            )


@pytest.fixture(scope="session")
def functional_mps(request):
    if not torch.backends.mps.is_available():
        reason = "PyTorch reports no available MPS GPU"
        _report_mps_preflight(request.config, "unavailable", reason)
        pytest.skip(reason)

    allocated_before = torch.mps.current_allocated_memory()
    try:
        sample = torch.ones(1, device="mps")
    except RuntimeError as error:
        if not hosted_mps_first_allocation_unavailable(
            error,
            allocated_before,
            sys.platform,
            os.environ.get("GITHUB_ACTIONS"),
            os.environ.get("RUNNER_ENVIRONMENT"),
        ):
            raise
        reason = (
            "GitHub-hosted macOS GPU unavailable: the first one-element MPS "
            "allocation failed with zero MPS bytes allocated"
        )
        detail = str(error).partition(" Use PYTORCH_MPS_HIGH_WATERMARK_RATIO")[0]
        _report_mps_preflight(request.config, "unavailable", f"{reason}. {detail}")
        pytest.skip(reason)

    # Errors after the first allocation, including arithmetic or synchronization,
    # remain failures on every host. Physical Mac allocation errors also propagate.
    result = sample + 1
    torch.mps.synchronize()
    assert result.cpu().item() == 2.0
    _report_mps_preflight(
        request.config,
        "available",
        "Bare Torch allocation, arithmetic and synchronization passed",
    )


@pytest.fixture(autouse=True)
def require_mps_hardware(request):
    if request.node.get_closest_marker("mps") is not None:
        request.getfixturevalue("functional_mps")
