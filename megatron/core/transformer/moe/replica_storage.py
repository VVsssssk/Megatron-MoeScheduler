# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Backend-independent replica consumer fences and exact directional caches."""

import weakref
from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaTransferHandle,
    ReplicaTransportConfig,
)


class ReplicaConsumerFence:
    """Serialize runtime consumers, including shared native-gradient staging.

    A transfer event only protects a reader from its producer. This fence
    protects the next producer from the previous GEMM/gradient consumer.
    """

    def __init__(self):
        self.owner = None
        self.done = None

    def acquire(self, owner, device):
        if self.owner is not None:
            raise RuntimeError("Replica workspace still has an active compute consumer.")
        if self.done is not None:
            torch.cuda.current_stream(device).wait_event(self.done)
        self.owner = owner

    def release(self, owner, device):
        if self.owner is not owner:
            raise RuntimeError("Replica workspace released by a different runtime.")
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(device))
        self.owner = None


_fences = weakref.WeakValueDictionary()


def get_replica_consumer_fence(group, device):
    """Native staging is shared even when replica storage is layer-local."""
    key = (group, device)
    fence = _fences.get(key)
    if fence is None:
        fence = ReplicaConsumerFence()
        _fences[key] = fence
    return fence


@dataclass
class _CacheEntry:
    placement: torch.Tensor
    sources: tuple
    handle: object


class ReplicaWeightCache:
    """Exact source snapshots catch writes that bypass Tensor version counters.

    Only private directional storage may opt in. A miss anywhere in the EP
    group makes every rank participate, including owners with no local slots.
    Snapshots cost one copy of the local source bytes per cached direction.
    """

    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.entries = {}
        self.hits = 0
        self.misses = 0
        self._miss = torch.zeros((), dtype=torch.int32, device=device)

    @staticmethod
    def tensors(sources):
        return tuple(t for source in sources for t in (*source.data, *(source.scales or ())))

    def lookup(self, layout, placement, sources):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Replica weight caching requires uncaptured host decisions.")
        entry = self.entries.get(layout)
        tensors = self.tensors(sources)
        hit = entry is not None and torch.equal(entry.placement, placement)
        hit = hit and len(entry.sources) == len(tensors)
        if hit:
            hit = all(
                old.dtype == new.dtype and old.shape == new.shape and torch.equal(old, new)
                for old, new in zip(entry.sources, tensors)
            )
        self._miss.fill_(not hit)
        dist.all_reduce(self._miss, op=dist.ReduceOp.MAX, group=self.group)
        if self._miss.item():
            self.misses += 1
            return None
        self.hits += 1
        return entry.handle

    def remember(self, layout, placement, sources, handle):
        # The backend retains in-flight sources. A cache hit only needs the
        # completion, not packed sends/receives from an already finished copy.
        completion = ReplicaTransferHandle(handle.transport, handle.completion)
        tensors = self.tensors(sources)
        old = self.entries.get(layout)
        if (
            old is not None
            and len(old.sources) == len(tensors)
            and all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(old.sources, tensors))
        ):
            old.placement.copy_(placement)
            for target, source in zip(old.sources, tensors):
                target.copy_(source)
            old.handle = completion
        else:
            self.entries[layout] = _CacheEntry(
                placement.clone(), tuple(t.clone() for t in tensors), completion
            )

    def clear(self):
        self.entries.clear()


class ReplicaNativeGradStorage:
    """Native gradient staging, optionally shared by serialized layer runtimes."""

    def __init__(self, config: ReplicaTransportConfig) -> None:
        self.buffers = tuple(
            torch.empty(
                (config.num_local_home_experts, *shape),
                dtype=config.grad_dtype,
                device=config.device,
            )
            for shape in config.member_shapes
        )


_native_grad_storage: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def get_native_grad_storage(config: ReplicaTransportConfig) -> ReplicaNativeGradStorage:
    if not config.share_native_grad_storage:
        return ReplicaNativeGradStorage(config)
    key = (
        config.group,
        config.device,
        config.num_local_home_experts,
        config.member_shapes,
        config.grad_dtype,
    )
    storage = _native_grad_storage.get(key)
    if storage is None:
        storage = ReplicaNativeGradStorage(config)
        _native_grad_storage[key] = storage
    return storage


