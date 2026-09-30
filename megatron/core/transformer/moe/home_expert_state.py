# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer storage adapters for physical home exchange.

Migration reconstructs one layer's authoritative state over expert DP before
NCCL exchanges it across EP. Destination views select their own shard ranges;
source and destination bucket boundaries need not match. This deliberately
trades temporary memory/communication for a simple first implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.optimizer.optimizer import OptimizerStateBinding

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.transformer.moe.replica_weight_transport import ReplicaTransportConfig


def optimizer_leaves(
    optimizer: MegatronOptimizer | torch.optim.Optimizer,
) -> list[MegatronOptimizer | torch.optim.Optimizer]:
    """Return leaf Megatron optimizers without depending on a particular wrapper."""
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return [optimizer]
    return [leaf for child in children for leaf in optimizer_leaves(child)]


def validate_home_expert_optimizer_config(config: OptimizerConfig | None) -> None:
    """Reject unsupported state representations before constructing an optimizer."""
    if getattr(config, "optimizer", "adam") not in ("adam", "sgd"):
        raise ValueError("Home exchange currently supports Adam and SGD optimizer state only.")
    if any(
        getattr(config, name, False)
        for name in (
            "use_precision_aware_optimizer",
            "use_layer_wise_distributed_optimizer",
            "optimizer_cpu_offload",
            "chunked_optimizer_state_offload",
            "offload_optimizer_states",
        )
    ):
        raise ValueError("Home exchange needs resident uncompressed optimizer state.")


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


@dataclass
class HomeStateShard:
    """A writable slice of one physical expert's authoritative state."""

    tensor: torch.Tensor | None
    start: int
    numel: int


class ShardedHomeExpertState:
    """Exchange adapter for typed, possibly uneven or empty physical state shards.

    ``components`` returns component-name -> H shard descriptors. Every rank
    exposes the same component names. Replicated states use ``dp_group=None``;
    sharded states must partition each expert exactly once over ``dp_group``.
    No Parameter, optimizer tensor or bucket view is replaced during install.
    """

    def __init__(
        self,
        components: Callable[[], dict[str, list[HomeStateShard]]],
        *,
        device: torch.device,
        dp_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.components = components
        self.device = device
        self.dp_group = dp_group
        self._destinations = None

    @torch.no_grad()
    def snapshot(self) -> tuple[torch.Tensor, ...]:
        components = self.components()
        names = sorted(components)
        self.component_names = tuple(names)
        metadata = {
            name: [
                (
                    s.start,
                    s.tensor.numel() if s.tensor is not None else 0,
                    s.numel,
                    s.tensor.dtype if s.tensor is not None else None,
                )
                for s in components[name]
            ]
            for name in names
        }
        if self.dp_group is None:
            gathered = [metadata]
        else:
            gathered = [None] * dist.get_world_size(self.dp_group)
            dist.all_gather_object(gathered, metadata, group=self.dp_group)
        if any(sorted(item) != names for item in gathered):
            raise ValueError("Home optimizer component schemas differ across expert DP ranks.")
        sources = []
        destinations = []
        for name in names:
            shards = components[name]
            full = []
            for slot, shard in enumerate(shards):
                descriptions = [item[name][slot] for item in gathered]
                dtypes = {dtype for _, size, _, dtype in descriptions if size}
                if len(dtypes) != 1 or any(n != shard.numel for _, _, n, _ in descriptions):
                    raise ValueError("Home state dtype/shape differs across shards.")
                end = 0
                for start, size, _, _ in sorted(descriptions, key=lambda row: row[0]):
                    if size:
                        if start != end:
                            raise ValueError("Home state shards overlap or leave a gap.")
                        end += size
                if end != shard.numel:
                    raise ValueError("Home state shards do not cover the full expert.")
                tensor = torch.zeros(shard.numel, dtype=next(iter(dtypes)), device=self.device)
                if shard.tensor is not None and shard.tensor.numel():
                    tensor[shard.start : shard.start + shard.tensor.numel()].copy_(
                        shard.tensor.detach().reshape(-1)
                    )
                if self.dp_group is not None:
                    dist.all_reduce(tensor, group=self.dp_group)
                full.append(tensor)
            sources.append(torch.stack(full))
            destinations.append(shards)
        self._destinations = destinations
        return tuple(sources)

    @torch.no_grad()
    def install(self, received: Sequence[torch.Tensor]) -> None:
        if self._destinations is None or len(received) != len(self._destinations):
            raise RuntimeError("Home state installation has no matching snapshot.")
        for source, shards in zip(received, self._destinations):
            for slot, shard in enumerate(shards):
                if shard.tensor is not None and shard.tensor.numel():
                    shard.tensor.copy_(
                        source[slot, shard.start : shard.start + shard.tensor.numel()].view(
                            shard.tensor.shape
                        )
                    )
        self._destinations = None


class OptimizerHomeExpertState(ShardedHomeExpertState):
    """Expose masters and tensor-valued optimizer states for DDP/DistOpt/MCore-FSDP.

    The optimizer must use ordinary full-precision state storage. Precision-aware
    compressed optimizer state and optimizer offload need dedicated adapters;
    they are rejected rather than silently dropping scale or residency metadata.
    Native MXFP8 model weights are regenerated using the existing optimizer cast
    path after all home layers have migrated.
    """

    def __init__(
        self,
        optimizer: MegatronOptimizer | torch.optim.Optimizer,
        source_parameters: Sequence[torch.Tensor],
        transport_config: ReplicaTransportConfig,
    ) -> None:
        self.transport_config = transport_config
        self.leaves = optimizer_leaves(optimizer)
        self._bindings: list[OptimizerStateBinding] = []
        dp_group = None
        for parameter in source_parameters:
            binding = self._find_binding(parameter)
            self._bindings.append(binding)
            group = binding[3]
            if group is not None:
                if dp_group is not None and group is not dp_group:
                    raise ValueError("One home layer cannot span different optimizer shard groups.")
                dp_group = group
        super().__init__(self._components, device=transport_config.device, dp_group=dp_group)
        # Check group options and any restored tensor-state schema now. Lazy
        # Adam/SGD moments may be absent until the first successful update.
        self._components()

    def _find_binding(self, parameter: torch.Tensor) -> OptimizerStateBinding:
        for leaf in self.leaves:
            config = getattr(leaf, "config", None)
            validate_home_expert_optimizer_config(config)
            if isinstance(leaf, torch.optim.Optimizer):
                if any(p is parameter for group in leaf.param_groups for p in group["params"]):
                    return OptimizerStateBinding(leaf, parameter, 0, None)
            else:
                binding = leaf.get_parameter_state_binding(parameter)
                if binding is not None:
                    return binding
        raise ValueError("Native expert is not owned by a supported optimizer state adapter.")

    def _components(self) -> dict[str, list[HomeStateShard]]:
        h = self.transport_config.num_local_home_experts
        steps = []
        for optim, main, _, _ in self._bindings:
            if main is not None and "step" in optim.state[main]:
                step = optim.state[main]["step"]
                steps.append(step.item() if isinstance(step, torch.Tensor) else step)
        # MCore assumes one common optimizer step. Verify it before leaving the
        # scalar in its fixed slot, including replicated and sharded optimizers.
        for group in (self.dp_group, self.transport_config.group):
            if group is not None:
                gathered_steps = [None] * dist.get_world_size(group)
                dist.all_gather_object(gathered_steps, steps, group=group)
                steps = [step for rank_steps in gathered_steps for step in rank_steps]
        if steps and any(step != steps[0] for step in steps):
            raise ValueError("Home migration requires a common per-parameter optimizer step.")
        option_keys = (
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "momentum",
            "dampening",
            "nesterov",
            "amsgrad",
            "maximize",
            "step",
        )
        options = [[], []]
        for index, (optim, main, _, _) in enumerate(self._bindings):
            if main is None:
                continue
            group = next(
                group
                for group in optim.param_groups
                if any(parameter is main for parameter in group["params"])
            )
            values = {
                key: (value.item() if isinstance(value, torch.Tensor) else value)
                for key in option_keys
                if (value := group.get(key)) is not None
            }
            options[index // h].append((type(optim).__module__, type(optim).__name__, values))
        for group in (self.dp_group, self.transport_config.group):
            if group is not None:
                gathered_options = [None] * dist.get_world_size(group)
                dist.all_gather_object(gathered_options, options, group=group)
                options = [
                    [
                        value
                        for rank_options in gathered_options
                        for value in rank_options[projection]
                    ]
                    for projection in range(2)
                ]
        for projection_options in options:
            if projection_options and any(
                value != projection_options[0] for value in projection_options
            ):
                raise ValueError(
                    "Home experts must have identical optimizer options within each projection."
                )
        components = {}
        for projection, shape in enumerate(self.transport_config.member_shapes):
            bindings = self._bindings[projection * h : (projection + 1) * h]
            # Missing shards still join schema discovery. State values are read
            # here, AFTER optimizer.step, rather than when the plan was staged.
            schemas = []
            states = []
            for optim, main, start, _ in bindings:
                state = {} if main is None else dict(optim.state[main])
                for key, value in state.items():
                    if key == "step":
                        continue  # MCore requires a common per-parameter step.
                    if not isinstance(value, torch.Tensor):
                        raise NotImplementedError(f"Unsupported per-expert optimizer state {key}.")
                    if _local(value).numel() != _local(main).numel():
                        raise NotImplementedError(
                            f"Unsupported non-elementwise optimizer state {key}."
                        )
                state = {key: _local(value) for key, value in state.items() if key != "step"}
                if main is not None:
                    state["master"] = _local(main)
                states.append(state)
                schemas.append(tuple(sorted(state)))
            gathered = [schemas]
            if self.dp_group is not None:
                gathered = [None] * dist.get_world_size(self.dp_group)
                dist.all_gather_object(gathered, schemas, group=self.dp_group)
            keys = set(key for rank in gathered for schema in rank for key in schema)
            if not keys or "master" not in keys:
                raise ValueError("No authoritative home master weights found.")
            for state, (_, main, _, _) in zip(states, bindings):
                if main is not None and _local(main).numel() and set(state) != keys:
                    raise ValueError("Home experts have different initialized optimizer states.")
            for key in sorted(keys):
                components[f"{projection}/{key}"] = [
                    HomeStateShard(state.get(key), binding[2], math.prod(shape))
                    for state, binding in zip(states, bindings)
                ]
        return components
