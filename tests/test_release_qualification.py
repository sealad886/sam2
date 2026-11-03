# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess

import pytest
import tools.qualify_release as qualification
import torch


@pytest.fixture
def simulated_gate(monkeypatch, tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    checkpoint = repository / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        qualification,
        "source_revision",
        lambda _: {"commit": "revision", "tree": "tree"},
    )
    monkeypatch.setattr(
        qualification, "physical_mps", lambda: {"mps_preflight": "passed"}
    )
    monkeypatch.setattr(
        qualification, "environment", lambda _: {"checkpoint_sha256": "checkpoint"}
    )
    monkeypatch.setattr(
        qualification.importlib.metadata, "version", lambda _: "test-version"
    )

    def run(arguments, **kwargs):
        if "pytest" in arguments:
            path = arguments[arguments.index("--junitxml") + 1]
            qualification.Path(path).write_text(
                '<testsuites><testsuite><testcase name="native"/>'
                '<testcase name="cuda"><skipped message="CUDA hardware unavailable"/>'
                "</testcase></testsuite></testsuites>"
            )
        else:
            path = arguments[arguments.index("--output") + 1]
            qualification.Path(path).write_text(
                json.dumps(
                    {
                        "passed": True,
                        "device": "mps",
                        "reference_device": "cpu",
                        "environment": {"checkpoint_sha256": "checkpoint"},
                        "gates": {"quality": True},
                        "agreement": {"image_iou": 1.0},
                        "live_memory_growth_bytes": 0,
                    }
                )
            )
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    return repository, checkpoint, output, run


def test_success_records_source_checkpoint_tests_and_quality(simulated_gate):
    repository, checkpoint, output, _ = simulated_gate
    receipt = qualification.qualify(checkpoint, output, repository)
    assert receipt["status"] == "passed"
    assert receipt["source"]["commit"] == "revision"
    assert receipt["environment"]["checkpoint_sha256"] == "checkpoint"
    assert receipt["tests"]["passed"] == 1
    assert len(receipt["tests"]["cuda_hardware_skips"]) == 1
    assert receipt["inference"]["agreement"]["image_iou"] == 1.0
    assert json.loads((output / "receipt.json").read_text()) == receipt


def test_inherited_pytest_selection_cannot_shorten_qualification(
    monkeypatch, simulated_gate
):
    repository, checkpoint, output, run = simulated_gate
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k selected_subset")
    environments = {}

    def observe_environment(arguments, **kwargs):
        stage = "tests" if "pytest" in arguments else "inference"
        environments[stage] = kwargs["env"]
        return run(arguments, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "run", observe_environment)
    receipt = qualification.qualify(checkpoint, output, repository)
    assert receipt["status"] == "passed"
    assert "PYTEST_ADDOPTS" not in environments["tests"]
    assert environments["inference"]["PYTEST_ADDOPTS"] == "-k selected_subset"


@pytest.mark.parametrize("stage", ["pytest", "tools.validate_inference"])
def test_subprocess_failure_cannot_leave_a_passing_receipt(
    monkeypatch, simulated_gate, stage
):
    repository, checkpoint, output, run = simulated_gate

    def fail(arguments, **kwargs):
        if stage in arguments:
            raise subprocess.CalledProcessError(1, arguments)
        return run(arguments, **kwargs)

    monkeypatch.setattr(qualification.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        qualification.qualify(checkpoint, output, repository)
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"


def test_source_change_invalidates_qualification(monkeypatch, simulated_gate):
    repository, checkpoint, output, _ = simulated_gate
    revisions = iter([{"commit": "before"}, {"commit": "after"}])
    monkeypatch.setattr(qualification, "source_revision", lambda _: next(revisions))
    with pytest.raises(RuntimeError, match="Source revision changed"):
        qualification.qualify(checkpoint, output, repository)
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("device", "cpu"),
        ("gates", {"quality": False}),
        ("environment", {"checkpoint_sha256": "different"}),
    ],
)
def test_failed_or_mismatched_inference_cannot_qualify_release(
    monkeypatch, simulated_gate, field, value
):
    repository, checkpoint, output, run = simulated_gate

    def alter_result(arguments, **kwargs):
        result = run(arguments, **kwargs)
        if "tools.validate_inference" in arguments:
            path = output / "inference.json"
            data = json.loads(path.read_text())
            data[field] = value
            path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(qualification.subprocess, "run", alter_result)
    with pytest.raises(RuntimeError, match="did not pass"):
        qualification.qualify(checkpoint, output, repository)
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"


def test_receipts_cannot_modify_source_or_overwrite_prior_evidence(simulated_gate):
    repository, checkpoint, output, _ = simulated_gate
    with pytest.raises(ValueError, match="outside"):
        qualification.qualify(checkpoint, repository / "evidence", repository)
    output.mkdir()
    prior = output / "receipt.json"
    prior.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        qualification.qualify(checkpoint, output, repository)
    assert prior.read_text() == "previous evidence"


@pytest.mark.parametrize(
    "reason", ["MPS unavailable", "could not import av", "optional dependency missing"]
)
def test_non_cuda_skips_cannot_qualify_a_release(tmp_path, reason):
    report = tmp_path / "tests.xml"
    report.write_text(
        f'<testsuites><testsuite><testcase><skipped message="{reason}"/></testcase></testsuite></testsuites>'
    )
    with pytest.raises(RuntimeError, match="skipped test"):
        qualification.test_summary(report)


def test_empty_test_report_cannot_qualify_a_release(tmp_path):
    report = tmp_path / "tests.xml"
    report.write_text("<testsuites/>")
    with pytest.raises(RuntimeError, match="no test results"):
        qualification.test_summary(report)


@pytest.fixture
def simulated_mac(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(qualification.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        qualification.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {"SPDisplaysDataType": [{"sppci_model": "Apple M3 Pro"}]}
        ),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", lambda *a, **k: torch.tensor([1.0]))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)


def test_reported_physical_mac_requires_functional_mps(simulated_mac):
    assert qualification.physical_mps()["mps_preflight"] == "passed"


@pytest.mark.parametrize("fallback", ["0", "1"])
def test_fallback_must_be_unset_even_when_disabled(
    monkeypatch, simulated_mac, fallback
):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", fallback)
    with pytest.raises(RuntimeError, match="Unset"):
        qualification.physical_mps()


def test_virtual_gpu_cannot_supply_physical_mac_evidence(monkeypatch, simulated_mac):
    monkeypatch.setattr(
        qualification.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {"SPDisplaysDataType": [{"sppci_model": "Apple Paravirtual device"}]}
        ),
    )
    with pytest.raises(RuntimeError, match="physical Apple M-series"):
        qualification.physical_mps()


def test_unavailable_mps_cannot_qualify_release(monkeypatch, simulated_mac):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS must be available"):
        qualification.physical_mps()


def test_source_requires_commit_and_ignores_finder_metadata(tmp_path):
    def git(*arguments):
        subprocess.run(
            ["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True
        )

    git("init", "-q")
    git(
        "-c",
        "user.name=Qualification test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "test fixture",
    )
    (tmp_path / ".DS_Store").write_bytes(b"Finder metadata")
    assert qualification.source_revision(tmp_path)["commit"]
    (tmp_path / "uncommitted.py").write_text("pass\n")
    with pytest.raises(RuntimeError, match="clean committed checkout"):
        qualification.source_revision(tmp_path)
