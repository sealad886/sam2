# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from training.utils.distributed import all_gather, broadcast_object


def _assert_payload(payload, rank):
    assert payload["rank"] == rank
    np.testing.assert_array_equal(payload["values"], np.arange(rank + 1))


def _exchange_python_objects(rank, rendezvous_path, result_dir):
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous_path).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        payload = {"rank": rank, "values": np.arange(rank + 1)}
        gathered = all_gather(payload, force_cpu=True)
        for source_rank, item in enumerate(gathered):
            _assert_payload(item, source_rank)
        gathered = all_gather(payload, force_filesys=True, filesys_save_dir=result_dir)
        for source_rank, item in enumerate(gathered):
            _assert_payload(item, source_rank)
        for use_disk in (False, True):
            received = broadcast_object(
                payload if rank == 1 else None, src=1, use_disk=use_disk
            )
            _assert_payload(received, 1)
        Path(result_dir, f"rank-{rank}.ok").write_text("all object exchanges passed\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
def test_two_rank_cpu_object_exchange(tmp_path):
    mp.spawn(
        _exchange_python_objects,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert sorted(path.name for path in tmp_path.glob("rank-*.ok")) == [
        "rank-0.ok",
        "rank-1.ok",
    ]
