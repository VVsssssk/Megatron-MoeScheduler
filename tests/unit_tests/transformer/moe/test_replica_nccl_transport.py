# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NCCL replica transport parity, subgroup routing and completion lifetime tests."""

import gc
import os
import socket
import weakref
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport, _compile_schedule
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaTransferHandle,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    create_replica_weight_transport,
)

from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


def test_host_schedule_matches_every_sender_and_receiver():
    table = [[2, 0, 2, -1], [0, 3, 4, 0], [-1, 1, 5, 2], [-1, -1, -1, -1]]
    plans = [_compile_schedule(table, 2, rank) for rank in range(4)]
    observed = []
    for owner, plan in enumerate(plans):
        for route in plan.sends:
            receiver = next(r for r in plans[route.peer].receives if r.peer == owner)
            assert receiver.home_indices == route.home_indices
            assert receiver.replica_slots == route.replica_slots
            observed.extend(
                (route.peer, slot, owner * 2 + index)
                for index, slot in zip(route.home_indices, route.replica_slots)
            )
        observed.extend(
            (owner, slot, owner * 2 + index)
            for index, slot in zip(plan.local.home_indices, plan.local.replica_slots)
        )
    expected = [
        (rank, slot, expert)
        for rank, row in enumerate(table)
        for slot, expert in enumerate(row)
        if expert != -1
    ]
    assert sorted(observed) == sorted(expected)
    assert not plans[3].sends and not plans[3].receives and not plans[3].local.replica_slots


@pytest.mark.parametrize("expert", [-2, 8])
def test_host_schedule_rejects_out_of_range_experts(expert):
    with pytest.raises(ValueError, match="expert id"):
        _compile_schedule([[expert], [-1], [-1], [-1]], 2, 0)


@pytest.fixture(scope="module")
def distributed_world():
    if not torch.cuda.is_available():
        pytest.skip("NCCL transport tests require CUDA.")
    # Utils defaults to LOCAL_RANK for its single-node tests. Supply the
    # global torchrun coordinates explicitly when spanning multiple nodes.
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


def test_distributed_launch_topology(distributed_world):
    rank = dist.get_rank()
    assert rank == int(os.environ["RANK"])
    assert dist.get_world_size() == int(os.environ["WORLD_SIZE"])
    assert torch.cuda.current_device() == int(os.environ["LOCAL_RANK"])
    members = [None] * dist.get_world_size()
    dist.all_gather_object(members, (rank, socket.gethostname(), torch.cuda.current_device()))
    assert len({(host, device) for _, host, device in members}) == len(members)
    if "NUM_NODES" in os.environ:
        assert len({host for _, host, _ in members}) == int(os.environ["NUM_NODES"])
    print(f"NCCL transport topology: {members}", flush=True)


@pytest.fixture(params=[1, 2, 4, 8])
def ep_group(request, distributed_world):
    world_size = dist.get_world_size()
    ep_size = request.param
    if world_size < ep_size or world_size % ep_size:
        pytest.skip(f"Requires a world size divisible by EP={ep_size}.")
    # Strided subgroups exercise global-rank conversion, including groups whose
    # first global rank is not zero. Every world member creates groups in order.
    count = world_size // ep_size
    group = None
    for offset in range(count):
        ranks = list(range(offset, world_size, count))
        candidate = dist.new_group(ranks, backend="nccl", timeout=timedelta(seconds=60))
        if dist.get_rank() in ranks:
            group = candidate
    try:
        yield group
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        dist.destroy_process_group(group)
        dist.barrier()


def _config(group, dtype=torch.float32):
    return ReplicaTransportConfig(
        group=group,
        device=torch.device("cuda", torch.cuda.current_device()),
        world_size=dist.get_world_size(group),
        num_local_home_experts=2,
        num_local_replica_slots=4,
        member_shapes=((8, 16), (16, 4)),
        weight_format="bf16",
        rowwise_scale_shapes=None,
        columnwise_scale_shapes=None,
        grad_dtype=dtype,
        num_sms=None,
    )


def _table(world_size, pattern):
    if pattern == "mixed":
        return [
            [2 * ((rank + 1) % world_size), 2 * rank, 2 * ((rank + 1) % world_size), -1]
            for rank in range(world_size)
        ]
    if pattern == "local":
        return [[2 * rank, 2 * rank + 1, 2 * rank, -1] for rank in range(world_size)]
    table = [[-1] * 4 for _ in range(world_size)]
    if pattern == "sparse":
        table[min(1, world_size - 1)][0] = 0
    return table


def _placement(config, table, version=1):
    return ReplicaPlacement(
        torch.tensor(table, dtype=torch.int32, device=config.device), ReplicaOwnership(2), version
    )


def _weight(config, projection, expert, offset=0):
    shape = config.member_shapes[projection]
    return (
        torch.arange(shape[0] * shape[1], device=config.device).reshape(shape) % 7
        + expert * 8
        + projection * 64
        + offset
    ).to(torch.bfloat16)


def _sources(config, rank, offset=0):
    return tuple(
        ReplicaWeightSource(
            tuple(_weight(config, p, 2 * rank + e, offset) for e in range(2)), scales=None
        )
        for p in range(2)
    )


def _replica_grad(config, projection, rank, slot):
    shape = config.member_shapes[projection]
    return (
        torch.arange(shape[0] * shape[1], device=config.device).reshape(shape) % 4 / 16
        + 1
        + rank / 8
        + slot / 16
        + projection / 4
    ).to(config.grad_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["mixed", "local", "empty", "sparse"])
def test_nccl_weights_and_gradients_match_reference(ep_group, dtype, pattern):
    config = _config(ep_group, dtype)
    rank = dist.get_rank(ep_group)
    table = _table(config.world_size, pattern)
    transport = create_replica_weight_transport("replica_nccl", config)
    try:
        assert isinstance(transport, NcclP2PTransport)
        plan = transport.prepare_plan(_placement(config, table))
        views = [transport.projection_views(p) for p in range(2)]
        pointers = [[w.data_ptr() for w in weights] for weights, _ in views]
        for weights, _ in views:
            for weight in weights:
                weight.fill_(-17)
        # A fresh source allocation on a different producer stream must be
        # visible on every consumer stream, including repeated waits.
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            sources = _sources(config, rank)
            handle = transport.start_weight_sync(sources=sources, plan=plan)
        for _ in range(2):
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                transport.wait_weight_sync(handle)
                copies = [tuple(w.clone() for w in weights) for weights, _ in views]
            consumer.synchronize()
            for p in range(2):
                for slot, expert in enumerate(table[rank]):
                    expected = (
                        _weight(config, p, expert)
                        if expert >= 0
                        else torch.full_like(copies[p][slot], -17)
                    )
                    torch.testing.assert_close(copies[p][slot], expected, rtol=0, atol=0)
        transport.wait_weight_sync(handle)
        # Use distinct gradient destinations to ensure start_grad_reduce honors
        # its argument rather than silently using transport-owned native staging.
        destinations = tuple(
            ReplicaGradDestination(
                tuple(torch.full_like(w, 256 + p * 16, dtype=dtype) for w in sources[p].data)
            )
            for p in range(2)
        )
        for p, (_, grads) in enumerate(views):
            transport.native_projection_grad_view(p).fill_(-999)
            for slot in range(4):
                grads[slot].copy_(_replica_grad(config, p, rank, slot))
                if table[rank][slot] < 0:
                    grads[slot].fill_(float("nan"))
        # Native expected values start nonzero. The reference directly sums the
        # global placement, independently of backend routes or packing offsets.
        expected = [
            [w.float().clone() for w in destination.tensors] for destination in destinations
        ]
        for source_rank, row in enumerate(table):
            for slot, expert in enumerate(row):
                if expert >= 0 and expert // 2 == rank:
                    for p in range(2):
                        expected[p][expert % 2].add_(
                            _replica_grad(config, p, source_rank, slot).float()
                        )
        fc2 = transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1,))
        transport.wait_grad_reduce(fc2)
        # FC1 must still contain native-only gradients until explicitly started.
        for tensor in destinations[0].tensors:
            torch.testing.assert_close(tensor, torch.full_like(tensor, 256), rtol=0, atol=0)
        fc1 = transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(0,))
        for _ in range(2):
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                transport.wait_grad_reduce(fc1)
                transport.wait_grad_reduce(fc2)
                copies = [[w.clone() for w in destination.tensors] for destination in destinations]
            consumer.synchronize()
            for p in range(2):
                for e in range(2):
                    torch.testing.assert_close(
                        copies[p][e], expected[p][e].to(dtype), rtol=0, atol=0
                    )
        for p in range(2):
            weights, _ = transport.projection_views(p)
            assert [w.data_ptr() for w in weights] == pointers[p]
            torch.testing.assert_close(
                transport.native_projection_grad_view(p),
                torch.full_like(transport.native_projection_grad_view(p), -999),
                rtol=0,
                atol=0,
            )
    finally:
        transport.destroy()
        transport.destroy()
    # Teardown must not destroy the caller's process group.
    dist.barrier(group=ep_group)


def test_nccl_old_plan_backward_uses_current_sources(ep_group):
    config = _config(ep_group)
    rank = dist.get_rank(ep_group)
    transport = NcclP2PTransport(config)
    try:
        table = _table(config.world_size, "mixed")
        first = transport.prepare_plan(_placement(config, table))
        second = transport.prepare_plan(_placement(config, _table(config.world_size, "local"), 2))
        for plan, offset in [(first, 0), (second, 8), (first, 16)]:
            handle = transport.start_weight_sync(sources=_sources(config, rank, offset), plan=plan)
            transport.wait_weight_sync(handle)
        for p in range(2):
            for slot, expert in enumerate(table[rank]):
                if expert >= 0:
                    torch.testing.assert_close(
                        transport.projection_views(p)[0][slot],
                        _weight(config, p, expert, 16),
                        rtol=0,
                        atol=0,
                    )
    finally:
        transport.destroy()


def test_nccl_dropped_handles_and_fp32_accumulation(ep_group):
    config = _config(ep_group, torch.bfloat16)
    rank = dist.get_rank(ep_group)
    transport = NcclP2PTransport(config)
    table = [[0] * 4 for _ in range(config.world_size)]
    try:
        plan = transport.prepare_plan(_placement(config, table))
        sources = _sources(config, rank)
        source_ref = weakref.ref(sources[0].data[0])
        weights = [transport.projection_views(p)[0] for p in range(2)]
        transport.start_weight_sync(sources=sources, plan=plan)
        del sources
        gc.collect()
        # Internal keepalive protects inputs even without a caller-owned handle.
        assert source_ref() is not None
        destinations = tuple(
            ReplicaGradDestination(tuple(transport.native_projection_grad_view(p)))
            for p in range(2)
        )
        for p in range(2):
            transport.native_projection_grad_view(p).fill_(256)
            transport.projection_views(p)[1].fill_(1)
        transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1, 0))
        # Drain without waiting on either handle. Adding individual ones into
        # BF16 256 would lose every contribution; FP32 accumulation retains them.
        transport.destroy()
        for p in range(2):
            for slot in range(4):
                torch.testing.assert_close(weights[p][slot], _weight(config, p, 0), rtol=0, atol=0)
            for index, tensor in enumerate(destinations[p].tensors):
                expected = 256 + (4 * config.world_size if rank == 0 and index == 0 else 0)
                torch.testing.assert_close(
                    tensor, torch.full_like(tensor, expected), rtol=0, atol=0
                )
    finally:
        transport.destroy()


def test_nccl_validation_and_capture_with_prepared_plan(ep_group, monkeypatch):
    config = _config(ep_group)
    with pytest.raises(ValueError, match="does not support mxfp8"):
        NcclP2PTransport(replace(config, weight_format="mxfp8"))
    with pytest.raises(ValueError, match="indexed CUDA"):
        NcclP2PTransport(replace(config, device=torch.device("cpu")))
    transport = NcclP2PTransport(config)
    try:
        placement = _placement(config, _table(config.world_size, "mixed"))
        plan = transport.prepare_plan(placement)
        sources = _sources(config, dist.get_rank(ep_group))
        with pytest.raises(ValueError, match="different transport"):
            transport.start_weight_sync(sources=sources, plan=replace(plan, transport=object()))
        with pytest.raises(ValueError, match="different transport"):
            transport.wait_weight_sync(ReplicaTransferHandle(object(), None))
        with pytest.raises(ValueError, match="plain BF16"):
            transport.start_weight_sync(
                sources=(replace(sources[0], layout=ReplicaWeightLayout.ROWWISE), sources[1]),
                plan=plan,
            )
        with pytest.raises(ValueError, match="canonical local expert"):
            transport.start_weight_sync(
                sources=(replace(sources[0], data=()), sources[1]), plan=plan
            )
        with pytest.raises(ValueError, match="dtype"):
            transport.start_weight_sync(
                sources=(
                    replace(sources[0], data=tuple(w.float() for w in sources[0].data)),
                    sources[1],
                ),
                plan=plan,
            )
        destinations = tuple(
            ReplicaGradDestination(tuple(transport.native_projection_grad_view(p)))
            for p in range(2)
        )
        with pytest.raises(ValueError, match="distinct"):
            transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(1, 1))
        # Mock the capture predicate so the rejection itself cannot invalidate
        # a real CUDA capture context. All operation entry points must reject,
        # including starts that reuse a schedule prepared outside capture.
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.prepare_plan(placement)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.start_weight_sync(sources=sources, plan=plan)
            with pytest.raises(ValueError, match="CUDA capture"):
                transport.start_grad_reduce(native_grads=destinations, plan=plan, projections=(0,))
    finally:
        transport.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        transport.projection_views(0)
