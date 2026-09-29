# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rebuilding input iterators must not advance checkpointed training RNG."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.training.datasets.data_samplers import build_pretraining_data_loader


def test_rebuilt_data_loaders_preserve_training_rng():
    args = SimpleNamespace(
        dataloader_type="single",
        micro_batch_size=1,
        full_validation=False,
        num_workers=0,
        use_varlen_dataset=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        seed=123,
    )
    module = "megatron.training.datasets.data_samplers"
    with (
        torch.random.fork_rng(devices=[]),
        patch(f"{module}.get_args", return_value=args),
        patch(f"{module}.mpu.get_data_parallel_rank", return_value=0),
        patch(f"{module}.mpu.get_data_parallel_world_size", return_value=1),
    ):
        torch.manual_seed(567)
        saved_rng = torch.get_rng_state().clone()
        # Reconstruct train/validation/test iterators after restoring RNG. Check
        # both fresh and resumed sampler positions, not a stand-in DataLoader.
        for consumed in (0, 4):
            for _ in range(3):
                loader = build_pretraining_data_loader(torch.arange(16), consumed)
                assert next(iter(loader)).item() == consumed
        assert torch.equal(saved_rng, torch.get_rng_state())
