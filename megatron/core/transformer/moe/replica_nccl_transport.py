# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed NCCL owner-to-replica weights and replica-to-owner gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaPlacement,
    ReplicaPreparedPlan,
    ReplicaTransferHandle,
    ReplicaTransportCapabilities,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    ReplicaWeightTransport,
)


@dataclass(frozen=True, slots=True)
class _PeerSlots:
    """One peer's entries, ordered by destination replica slot."""

    peer: int
    home_indices: tuple[int, ...]
    replica_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _NcclSchedule:
    """Weight-direction routes; gradients traverse the same routes backwards."""

    sends: tuple[_PeerSlots, ...]
    receives: tuple[_PeerSlots, ...]
    local: _PeerSlots


def _compile_schedule(table: list[list[int]], home_experts: int, rank: int) -> _NcclSchedule:
    """Compile an identical global slot table without retaining tensor addresses."""
    sends: dict[int, list[tuple[int, int]]] = {}
    receives: dict[int, list[tuple[int, int]]] = {}
    local = []
    num_experts = len(table) * home_experts
    for destination, row in enumerate(table):
        for slot, expert in enumerate(row):
            if expert == -1:
                continue
            if not 0 <= expert < num_experts:
                raise ValueError(f"Replica expert id {expert} is outside [0, {num_experts}).")
            owner, index = divmod(expert, home_experts)
            if owner == destination == rank:
                local.append((index, slot))
            elif owner == rank:
                sends.setdefault(destination, []).append((index, slot))
            elif destination == rank:
                receives.setdefault(owner, []).append((index, slot))

    def entries(peer, pairs):
        return _PeerSlots(
            peer, tuple(index for index, _ in pairs), tuple(slot for _, slot in pairs)
        )

    return _NcclSchedule(
        tuple(entries(peer, pairs) for peer, pairs in sorted(sends.items())),
        tuple(entries(peer, pairs) for peer, pairs in sorted(receives.items())),
        entries(rank, local),
    )


class NcclP2PTransport(ReplicaWeightTransport):
    """Transport BF16 expert storage over a borrowed NCCL EP process group.

    Construction is collective over ``config.group`` and must occur in the
    same order on every member. A small all-reduce initializes the group before
    sparse batched P2P (whose first use otherwise requires every rank). All
    ranks must supply identical placements and issue transport operations in
    matching order, including relative to other operations on the same group.

    Planning synchronously copies the slot table to the host once per plan.
    Dynamic host schedules are not CUDA-graph safe, including when prepared
    before capture. Storage and one communication stream are layer-local.
    Callers must finish consumers before overwriting reusable replica storage.
    ``num_sms`` is not a portable NCCL P2P launch control and is unused here.
    """

    transport_name = "replica_nccl"
    capabilities = ReplicaTransportCapabilities(
        weight_formats=("bf16",), grad_dtypes=(torch.bfloat16, torch.float32)
    )

    def __init__(self, config: ReplicaTransportConfig) -> None:
        super().__init__(config)
        if config.device.type != "cuda" or config.device.index is None:
            raise ValueError("replica_nccl requires an indexed CUDA device.")
        if config.group is None or not dist.is_initialized():
            raise ValueError("replica_nccl requires an initialized explicit NCCL process group.")
        if dist.get_backend(config.group) != "nccl":
            raise ValueError("replica_nccl requires a NCCL process group.")
        if dist.get_world_size(config.group) != config.world_size:
            raise ValueError("replica_nccl world_size does not match its process group.")
        if (
            config.num_local_home_experts <= 0
            or config.num_local_replica_slots < 0
            or len(config.member_shapes) != 2
            or any(len(shape) != 2 or min(shape) <= 0 for shape in config.member_shapes)
        ):
            raise ValueError("replica_nccl requires valid expert counts and two matrix shapes.")
        self._destroyed = False
        self._check_available()
        self.rank = dist.get_rank(config.group)
        self._global_ranks = tuple(
            dist.get_global_rank(config.group, rank) for rank in range(config.world_size)
        )
        self._numels = tuple(math.prod(shape) for shape in config.member_shapes)
        self._weights = tuple(
            torch.empty(
                (config.num_local_replica_slots, *shape), dtype=torch.bfloat16, device=config.device
            )
            for shape in config.member_shapes
        )
        self._replica_grads = tuple(
            torch.empty_like(weight, dtype=config.grad_dtype) for weight in self._weights
        )
        self._native_grads = tuple(
            torch.empty(
                (config.num_local_home_experts, *shape),
                dtype=config.grad_dtype,
                device=config.device,
            )
            for shape in config.member_shapes
        )
        self._stream = torch.cuda.Stream(device=config.device)
        self._inflight: list[tuple[torch.cuda.Event, tuple]] = []
        # All members participate even when this layer's eventual plan has no
        # traffic on some ranks. This group is borrowed, never destroyed here.
        ready = torch.zeros(1, device=config.device)
        dist.all_reduce(ready, group=config.group)

    def _check_available(self) -> None:
        if self._destroyed:
            raise RuntimeError("replica_nccl transport was destroyed.")
        if torch.cuda.current_device() != self.config.device.index:
            raise ValueError("Set the current CUDA device to the replica_nccl transport device.")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("replica_nccl does not support CUDA capture with host schedules.")

    def _projection_index(self, projection_index: int) -> None:
        if projection_index not in (0, 1):
            raise ValueError("Replica projection index must be 0 (FC1) or 1 (FC2).")
        if self._destroyed:
            raise RuntimeError("replica_nccl transport was destroyed.")

    @property
    def grad_dtype(self) -> torch.dtype:
        """Return the replica gradient storage and wire dtype."""
        return self.config.grad_dtype

    def projection_views(
        self, projection_index: int
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return stable BF16 replica weight views and gradient storage."""
        self._projection_index(projection_index)
        return tuple(self._weights[projection_index]), self._replica_grads[projection_index]

    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        """Return native staging; its producer, not the transport, initializes it."""
        self._projection_index(projection_index)
        return self._native_grads[projection_index]

    def prepare_plan(self, placement: ReplicaPlacement) -> ReplicaPreparedPlan:
        """Read device placement synchronously and compile deterministic peer lists."""
        self._check_available()
        super().prepare_plan(placement)
        schedule = _compile_schedule(
            placement.slot_to_expert.cpu().tolist(), placement.ownership.home_experts, self.rank
        )
        return ReplicaPreparedPlan(placement, self, schedule)

    def _schedule(self, plan: ReplicaPreparedPlan) -> _NcclSchedule:
        self._check_available()
        self.validate_plan(plan)
        if not isinstance(plan.metadata, _NcclSchedule):
            raise ValueError("replica_nccl requires a compiled NCCL schedule.")
        # Keep inputs alive even if the caller drops a handle before GPU completion.
        self._inflight = [(event, refs) for event, refs in self._inflight if not event.query()]
        return plan.metadata

    def _validate_tensors(self, tensors, projection: int, dtype: torch.dtype) -> None:
        if len(tensors) != self.config.num_local_home_experts:
            raise ValueError("Replica tensors must contain every canonical local expert.")
        for tensor in tensors:
            if (
                tensor.device != self.config.device
                or tensor.dtype != dtype
                or tensor.numel() != self._numels[projection]
                or not tensor.is_contiguous()
            ):
                raise ValueError("Replica tensor device, dtype, size or contiguity is invalid.")

    def _exchange(self, sends, receive_sizes, dtype):
        """Enqueue one packed message per directed peer pair on the current stream."""
        receives = {
            peer: torch.empty(size, dtype=dtype, device=self.config.device)
            for peer, size in receive_sizes.items()
        }
        # Sorting directed edges gives every rank a consistent subset of one
        # global ordering. NCCL has no tags to distinguish mismatched messages.
        edges = [(self.rank, peer, dist.isend, tensor) for peer, tensor in sends.items()]
        edges.extend((peer, self.rank, dist.irecv, tensor) for peer, tensor in receives.items())
        ops = [
            dist.P2POp(
                op,
                tensor,
                self._global_ranks[destination if source == self.rank else source],
                group=self.config.group,
            )
            for source, destination, op, tensor in sorted(edges, key=lambda edge: edge[:2])
        ]
        works = tuple(dist.batch_isend_irecv(ops)) if ops else ()
        for work in works:
            # Connect NCCL's internal stream to this stream before unpack/add.
            # Under NCCL blocking-wait settings this may also block the host.
            work.wait()
        return receives, (tuple(sends.values()), tuple(receives.values()), works)

    def _finish(self, *keepalive) -> ReplicaTransferHandle:
        done = torch.cuda.Event()
        done.record(self._stream)
        self._inflight.append((done, keepalive))
        return ReplicaTransferHandle(self, done, keepalive)

    @torch.no_grad()
    def start_weight_sync(
        self, *, sources: tuple[ReplicaWeightSource, ...], plan: ReplicaPreparedPlan
    ) -> ReplicaTransferHandle:
        """Pack FC1/FC2 by peer and complete only after local copies and unpack."""
        schedule = self._schedule(plan)
        if len(sources) != 2:
            raise ValueError("replica_nccl requires FC1 and FC2 weight sources.")
        for projection, source in enumerate(sources):
            if source.layout is not ReplicaWeightLayout.PLAIN or source.scales is not None:
                raise ValueError("replica_nccl supports plain BF16 weights without scales.")
            self._validate_tensors(source.data, projection, torch.bfloat16)
        self._stream.wait_stream(torch.cuda.current_stream(self.config.device))
        with torch.cuda.stream(self._stream):
            for source in sources:
                for tensor in source.data:
                    tensor.record_stream(self._stream)
            for weights in self._weights:
                weights.record_stream(self._stream)
            sends = {
                route.peer: torch.cat(
                    [
                        source.data[index].view(-1)
                        for source in sources
                        for index in route.home_indices
                    ]
                )
                for route in schedule.sends
            }
            receives, buffers = self._exchange(
                sends,
                {
                    route.peer: len(route.replica_slots) * sum(self._numels)
                    for route in schedule.receives
                },
                torch.bfloat16,
            )
            for projection, source in enumerate(sources):
                for index, slot in zip(schedule.local.home_indices, schedule.local.replica_slots):
                    self._weights[projection][slot].view(-1).copy_(source.data[index].view(-1))
            for route in schedule.receives:
                offset = 0
                for projection, numel in enumerate(self._numels):
                    for slot in route.replica_slots:
                        self._weights[projection][slot].view(-1).copy_(
                            receives[route.peer].narrow(0, offset, numel)
                        )
                        offset += numel
            return self._finish(plan, sources, buffers)

    def wait_weight_sync(self, handle: ReplicaTransferHandle) -> None:
        """Order every calling stream after transfer and postprocessing."""
        self._check_available()
        if handle.transport is not self:
            raise ValueError("Replica completion belongs to a different transport.")
        torch.cuda.current_stream(self.config.device).wait_event(handle.completion)

    @torch.no_grad()
    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        plan: ReplicaPreparedPlan,
        projections: tuple[int, ...],
    ) -> ReplicaTransferHandle:
        """Return selected replica gradients and add in FP32 with one final cast."""
        schedule = self._schedule(plan)
        if (
            not projections
            or len(set(projections)) != len(projections)
            or any(projection not in (0, 1) for projection in projections)
            or len(native_grads) != 2
        ):
            raise ValueError("Select distinct FC1/FC2 projections and supply both destinations.")
        projections = tuple(sorted(projections))
        for projection in projections:
            self._validate_tensors(native_grads[projection].tensors, projection, self.grad_dtype)
        self._stream.wait_stream(torch.cuda.current_stream(self.config.device))
        with torch.cuda.stream(self._stream):
            for projection in projections:
                self._replica_grads[projection].record_stream(self._stream)
                for tensor in native_grads[projection].tensors:
                    tensor.record_stream(self._stream)
            sends = {
                route.peer: torch.cat(
                    [
                        self._replica_grads[projection][slot].view(-1)
                        for projection in projections
                        for slot in route.replica_slots
                    ]
                )
                for route in schedule.receives
            }
            receives, buffers = self._exchange(
                sends,
                {
                    route.peer: len(route.home_indices) * sum(self._numels[p] for p in projections)
                    for route in schedule.sends
                },
                self.grad_dtype,
            )
            # Each accumulator includes the pre-existing native gradient. Do
            # not round back to BF16 after each replica or overwrite native GEMM output.
            accumulators = {}

            def accumulate(projection, index, contribution):
                key = (projection, index)
                if key not in accumulators:
                    accumulators[key] = (
                        native_grads[projection]
                        .tensors[index]
                        .view(-1)
                        .to(dtype=torch.float32, copy=True)
                    )
                accumulators[key].add_(contribution.float())

            for projection in projections:
                for index, slot in zip(schedule.local.home_indices, schedule.local.replica_slots):
                    accumulate(projection, index, self._replica_grads[projection][slot].view(-1))
            for route in schedule.sends:
                offset = 0
                for projection in projections:
                    numel = self._numels[projection]
                    for index in route.home_indices:
                        accumulate(projection, index, receives[route.peer].narrow(0, offset, numel))
                        offset += numel
            for (projection, index), accumulator in accumulators.items():
                native_grads[projection].tensors[index].view(-1).copy_(accumulator)
            return self._finish(plan, native_grads, buffers, accumulators)

    def wait_grad_reduce(self, handle: ReplicaTransferHandle) -> None:
        """Order the caller after communication, accumulation and final cast."""
        self.wait_weight_sync(handle)

    def destroy(self) -> None:
        """Drain this layer's operations and release storage, retaining the borrowed group."""
        if self._destroyed:
            return
        self._check_available()
        self._stream.synchronize()
        self._inflight.clear()
        self._weights = self._replica_grads = self._native_grads = ()
        self._destroyed = True
