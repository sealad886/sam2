# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_setup(monkeypatch):
    """Exercise setup decisions without requiring a CUDA toolkit or compiler."""
    captured = {}
    setuptools = ModuleType("setuptools")
    setuptools.find_packages = lambda **kwargs: ["sam2"]
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.chdir(ROOT)

    class BuildExtension:
        @classmethod
        def with_options(cls, **kwargs):
            return cls

    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = BuildExtension
    cpp_extension.CUDAExtension = lambda *args, **kwargs: object()
    original_import = builtins.__import__

    def run(platform, build_cuda=None, allow_errors="0", tools_available=True):
        monkeypatch.setattr(sys, "platform", platform)
        if build_cuda is None:
            monkeypatch.delenv("SAM2_BUILD_CUDA", raising=False)
        else:
            monkeypatch.setenv("SAM2_BUILD_CUDA", build_cuda)
        monkeypatch.setenv("SAM2_BUILD_ALLOW_ERRORS", allow_errors)

        def import_module(name, *args, **kwargs):
            if name == "torch.utils.cpp_extension":
                if not tools_available:
                    raise RuntimeError("CUDA build tools are unavailable")
                return cpp_extension
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_module)
        runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
        return captured

    return run


def test_mac_default_installs_without_cuda_build_tools(run_setup):
    package = run_setup("darwin", tools_available=False)
    assert package["ext_modules"] == []
    assert package["cmdclass"] == {}


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_disabling_cuda_needs_no_cuda_build_tools(run_setup, platform):
    package = run_setup(platform, build_cuda="0", tools_available=False)
    assert package["ext_modules"] == []
    assert package["cmdclass"] == {}


@pytest.mark.parametrize(
    ("platform", "build_cuda"), [("darwin", "1"), ("linux", None), ("win32", None)]
)
def test_requested_cuda_build_is_preserved(run_setup, platform, build_cuda):
    package = run_setup(platform, build_cuda=build_cuda)
    assert len(package["ext_modules"]) == 1
    assert "build_ext" in package["cmdclass"]


def test_requested_cuda_build_propagates_strict_errors(run_setup):
    with pytest.raises(RuntimeError, match="CUDA build tools are unavailable"):
        run_setup("darwin", build_cuda="1", tools_available=False)


def test_requested_cuda_build_can_allow_errors(run_setup):
    package = run_setup(
        "darwin", build_cuda="1", allow_errors="1", tools_available=False
    )
    assert package["ext_modules"] == []
    assert package["cmdclass"] == {}
