# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Non-GTP training regression for PR #7450 planner/replica alignment."""

import faulthandler
import os
import traceback
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.replica_expert_runtime import _ReplicaProjection
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_scheduler_bf16_layer_parity as parity

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def distributed_world():
    if not torch.cuda.is_available():
        pytest.skip("Replica training tests require CUDA.")
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


@pytest.fixture(autouse=True)
def report_stalled_test():
    faulthandler.dump_traceback_later(90, repeat=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("backend", ["replica_peer_tma", "replica_nccl"])
@pytest.mark.parametrize("direct", [False, True])
def test_bf16_microbatch_accumulation_and_adam(distributed_world, monkeypatch, backend, direct):
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    original = parity._config
    monkeypatch.setattr(
        parity,
        "_config",
        lambda backend, capacity=None: replace(
            original(backend, capacity),
            moe_scheduler_planner_type="moon_ep",
            moe_router_mock_maxvio=4,
        ),
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    g = torch.Generator(device="cuda").manual_seed(413 + dist.get_rank())
    inputs = [
        torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(4)
    ]
    upstream = [
        torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=g) * 0.01 for x in inputs
    ]
    bind = _ReplicaProjection.prepare_runtime_parameters
    seen = []

    def inspect(projection, dtype):
        # Exercise fallback with normal DDP hook semantics but overwrite staging.
        # Temporarily hide the DDP capability only while selecting its binding.
        saved = []
        if not direct:
            for p in projection.parameters:
                saved.append(p.grad_added_to_main_grad)
                del p.grad_added_to_main_grad
        try:
            bind(projection, dtype)
        finally:
            for p, value in zip(projection.parameters, saved):
                p.grad_added_to_main_grad = value
        seen.append(projection.accumulates_main_grad)
        assert projection.accumulates_main_grad == direct
        for runtime in projection.runtime_parameters:
            assert not hasattr(runtime, "grad_added_to_main_grad")

    try:
        initial, reference, _ = parity._run(None, pg, None, inputs, upstream)
        monkeypatch.setattr(_ReplicaProjection, "prepare_runtime_parameters", inspect)
        _, actual, dispatch = parity._run(backend, pg, initial, inputs, upstream)
        assert seen and not any(d["overflow"] for d in dispatch)
        failures = []
        for name in reference:
            m = parity._metrics(reference[name], actual[name])
            limit = 2.5e-2 if name.startswith("master_delta/") else 1e-2
            if not m["finite"] or m["relative_l2"] > limit:
                failures.append((name, m))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("routing", ["mock", "learned", "squared_relu"])
def test_precision_training_three_updates(distributed_world, monkeypatch, precision, routing):
    from megatron.core.activations import squared_relu
    from tests.unit_tests.transformer.moe import scheduler_precision_parity_utils as precision_utils
    from tests.unit_tests.transformer.moe.scheduler_precision_parity_utils import run_leg

    if routing == "squared_relu":
        original_config = precision_utils.config_for
        monkeypatch.setattr(
            precision_utils,
            "config_for",
            lambda *args, **kwargs: replace(
                original_config(*args, **kwargs),
                activation_func=squared_relu,
                gated_linear_unit=False,
                use_fused_weighted_squared_relu=True,
            ),
        )
        routing = "learned"
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    g = torch.Generator(device="cuda").manual_seed(837 + dist.get_rank())
    inputs = [
        [
            torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(4)
        ]
        for _ in range(3)
    ]
    upstream = [
        [torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=g) * 0.01 for x in xs]
        for xs in inputs
    ]
    try:
        reference, _ = run_leg(precision, pg, inputs, upstream, routing=routing)
        actual, diagnostics = run_leg(
            precision,
            pg,
            inputs,
            upstream,
            ("moon_ep", "peer_tma", "shared", "reuse"),
            routing=routing,
        )
        failures = []
        assert reference.keys() == actual.keys()
        for name in reference:
            if name.startswith("initial"):
                if not torch.equal(reference[name], actual[name]):
                    failures.append((name, "initialization differs"))
                continue
            m = parity._metrics(reference[name], actual[name])
            limit = 0.025 if "/master_delta/" in name else 0.01
            if not m["finite"] or m["relative_l2"] > limit:
                failures.append((name, m))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
        assert len(diagnostics) == 3
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        Utils.destroy_model_parallel()
