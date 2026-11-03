# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the maintainer's physical Apple Silicon release qualification checklist.

Run manually from a clean committed checkout, using its project environment:
env -u PYTORCH_ENABLE_MPS_FALLBACK python -m tools.qualify_release \
    --output-dir /tmp/sam2-release-qualification
This records evidence only; it never creates or publishes a release.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch
from tools.validate_inference import environment

REPOSITORY = Path(__file__).resolve().parents[1]


def source_revision(repository):
    """Require a committed checkout; ignore Finder metadata during discovery."""

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments], text=True
        ).strip()

    if git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude,glob)**/.DS_Store",
    ):
        raise RuntimeError("Qualification requires a clean committed checkout")
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")}


def physical_mps():
    """Reject virtual/unavailable GPUs before running the qualification workload."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is not None:
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK before qualification")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Qualification requires a physical Apple Silicon Mac")
    displays = json.loads(
        subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            text=True,
            timeout=30,
        )
    )
    models = [
        display.get("sppci_model", display.get("_name", ""))
        for display in displays.get("SPDisplaysDataType", [])
    ]
    if not any(re.match(r"^Apple M\d+(?:\s|$)", model) for model in models) or any(
        "virtual" in model.lower() for model in models
    ):
        raise RuntimeError(
            "No reported physical Apple M-series GPU; VM results are insufficient"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS must be available for release qualification")
    sample = torch.ones(1, device="mps") + 1
    torch.mps.synchronize()
    if sample.cpu().item() != 2:
        raise RuntimeError("MPS arithmetic preflight failed")
    return {"gpu_models": models, "mps_preflight": "passed", "fallback": "unset"}


def test_summary(path):
    """Allow unavailable CUDA hardware; no optional-dependency or MPS skips."""
    cases = ET.parse(path).getroot().findall(".//testcase")
    if not cases:
        raise RuntimeError("The test suite produced no test results")
    skipped = []
    for case in cases:
        if case.find("failure") is not None or case.find("error") is not None:
            raise RuntimeError("The test suite reported a failure")
        skip = case.find("skipped")
        if skip is not None:
            reason = skip.get("message", "")
            if not re.search(r"\bCUDA\b.*\bunavailable\b", reason, re.IGNORECASE):
                raise RuntimeError(
                    f"Release qualification cannot accept a skipped test: {reason}"
                )
            skipped.append({"test": case.get("name"), "reason": reason})
    return {"passed": len(cases) - len(skipped), "cuda_hardware_skips": skipped}


def qualify(checkpoint, output_dir, repository=REPOSITORY):
    repository, checkpoint, output_dir = (
        Path(path).resolve() for path in (repository, checkpoint, output_dir)
    )
    if output_dir.is_relative_to(repository):
        raise ValueError("Write qualification evidence outside the source checkout")
    # A unique directory prevents a failed run from leaving a stale passing receipt.
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "scope": "physical Apple Silicon; full tests and tiny-checkpoint CPU/MPS regression",
        "limitations": [
            "No release publication",
            "No CUDA qualification",
            "No dataset accuracy or long-video guarantee",
        ],
    }
    try:
        receipt["source"] = source_revision(repository)
        receipt["hardware"] = physical_mps()
        receipt["environment"] = environment(checkpoint)
        receipt["environment"].update(
            {name: importlib.metadata.version(name) for name in ("torchvision", "av")}
        )

        def run(stage, arguments):
            print(f"Release qualification: {stage}", flush=True)
            child_environment = os.environ.copy()
            if stage == "tests":
                # An inherited -k/-m selector must not turn this into a partial run.
                child_environment.pop("PYTEST_ADDOPTS", None)
            with (output_dir / f"{stage}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, *arguments],
                    cwd=repository,
                    env=child_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

        run(
            "tests",
            [
                "-m",
                "pytest",
                "-q",
                "-rs",
                "--junitxml",
                str(output_dir / "tests.xml"),
                "tests",
            ],
        )
        receipt["tests"] = test_summary(output_dir / "tests.xml")
        run(
            "inference",
            [
                "-m",
                "tools.validate_inference",
                "--device",
                "mps",
                "--reference-device",
                "cpu",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output_dir / "inference.json"),
            ],
        )
        inference = json.loads((output_dir / "inference.json").read_text())
        if (
            inference.get("passed") is not True
            or inference.get("device") != "mps"
            or inference.get("reference_device") != "cpu"
            or not inference.get("gates")
            or not all(value is True for value in inference["gates"].values())
            or inference["environment"]["checkpoint_sha256"]
            != receipt["environment"]["checkpoint_sha256"]
        ):
            raise RuntimeError("The CPU/MPS checkpoint qualification did not pass")
        if source_revision(repository) != receipt["source"]:
            raise RuntimeError("Source revision changed during qualification")
        if (
            environment(checkpoint)["checkpoint_sha256"]
            != receipt["environment"]["checkpoint_sha256"]
        ):
            raise RuntimeError("Checkpoint changed during qualification")
        receipt["inference"] = {
            "gates": inference["gates"],
            "agreement": inference["agreement"],
            "live_memory_growth_bytes": inference["live_memory_growth_bytes"],
        }
        receipt["status"] = "passed"
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY / "checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory outside the checkout",
    )
    args = parser.parse_args()
    try:
        qualify(args.checkpoint, args.output_dir)
    except Exception as error:
        print(f"Release qualification failed: {error}", file=sys.stderr)
        return 1
    print(f"Release qualification passed; evidence: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
