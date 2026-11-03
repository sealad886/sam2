# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.utils.checkpoint_utils import (
    CkptExcludeKernel,
    load_checkpoint,
    load_checkpoint_and_apply_kernels,
)


def _load_weights(path, use_kernels, **kwargs):
    if use_kernels:
        return load_checkpoint_and_apply_kernels(
            str(path),
            checkpoint_kernels=[CkptExcludeKernel(["unused"])],
            **kwargs,
        )
    return load_checkpoint([str(path)], pick_recursive_keys=["state_dict"], **kwargs)


@pytest.mark.parametrize("use_kernels", [False, True])
def test_checkpoint_weights_load_by_default(tmp_path, use_kernels):
    path = tmp_path / "weights.pt"
    weights = {"weight": torch.tensor([1.0, 2.0]), "unused": torch.zeros(1)}
    torch.save({"state_dict": weights, "epoch": 3}, path)

    loaded = _load_weights(path, use_kernels)

    torch.testing.assert_close(loaded["weight"], weights["weight"])
    if use_kernels:
        assert "unused" not in loaded


@pytest.mark.parametrize("use_kernels", [False, True])
def test_checkpoint_requires_opt_in_for_trusted_python_state(tmp_path, use_kernels):
    path = tmp_path / "trusted-training-state.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1), "unused": torch.zeros(1)},
            "metrics": np.array([0.5, 0.8]),
        },
        path,
    )

    with pytest.raises(pickle.UnpicklingError):
        _load_weights(path, use_kernels)

    loaded = _load_weights(path, use_kernels, weights_only=False)
    torch.testing.assert_close(loaded["weight"], torch.ones(1))
    trusted = load_checkpoint([str(path)], weights_only=False)
    np.testing.assert_array_equal(trusted["metrics"], [0.5, 0.8])


def _checkpoint_trainer(checkpoint_dir):
    # Trainer construction starts a whole training job. These tests exercise its
    # real checkpoint APIs with small CPU components and a real optimizer/scaler.
    pytest.importorskip("tensordict")
    pytest.importorskip("tensorboard")
    from training.trainer import CheckpointConf, OptimConf, Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optim = SimpleNamespace(
        optimizer=torch.optim.Adam(trainer.model.parameters(), lr=0.01)
    )
    trainer.optim_conf = OptimConf(amp={"enabled": True})
    trainer.scaler = torch.amp.GradScaler("cpu")
    trainer.loss = torch.nn.ModuleDict({"mse": torch.nn.MSELoss()})
    trainer.checkpoint_conf = CheckpointConf(save_dir=str(checkpoint_dir), save_freq=0)
    trainer.steps = {"train": 1, "val": 0}
    trainer.time_elapsed_meter = SimpleNamespace(val=12.5)
    trainer.best_meter_values = {"validation": np.float32(0.8)}
    trainer.distributed_rank = 0
    trainer.train_dataset = None
    return trainer


def _optimizer_update(trainer):
    trainer.optim.optimizer.zero_grad()
    loss = trainer.loss["mse"](
        trainer.model(torch.tensor([[1.0, 2.0], [2.0, 1.0]])),
        torch.tensor([[0.5], [1.0]]),
    )
    trainer.scaler.scale(loss).backward()
    trainer.scaler.step(trainer.optim.optimizer)
    trainer.scaler.update()


def test_trainer_resumes_full_state_and_next_optimizer_update(tmp_path):
    trainer = _checkpoint_trainer(tmp_path)
    _optimizer_update(trainer)
    trainer.save_checkpoint(epoch=3)

    resumed = _checkpoint_trainer(tmp_path)
    resumed.load_checkpoint()

    assert resumed.epoch == 3
    assert resumed.steps == trainer.steps
    assert resumed.ckpt_time_elapsed == 12.5
    assert resumed.best_meter_values == trainer.best_meter_values
    assert resumed.scaler.state_dict() == trainer.scaler.state_dict()
    assert resumed.optim.optimizer.state_dict()["state"]
    for expected, actual in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected)

    # Continuing the same step must produce the same parameters and Adam state.
    _optimizer_update(trainer)
    _optimizer_update(resumed)
    for expected, actual in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected)
    expected_state = trainer.optim.optimizer.state_dict()["state"]
    actual_state = resumed.optim.optimizer.state_dict()["state"]
    for index, values in expected_state.items():
        for name, expected in values.items():
            torch.testing.assert_close(actual_state[index][name], expected)
