# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-loop integration for the explicit MoEScheduler.step boundary."""

import torch

from megatron.core.transformer.moe.home_expert_state import optimizer_leaves
from megatron.core.transformer.moe.moe_scheduler import MoEScheduler


def validate_home_expert_checkpoint_args(args, checkpoint_args=None):
    """Reject formats or topology changes that lose physical expert ownership."""
    active = getattr(args, "moe_scheduler_home_update_interval", 0)
    saved = getattr(checkpoint_args, "moe_scheduler_home_update_interval", 0)
    if not active and not saved:
        return
    if args.ckpt_format not in ("torch_dist", "fsdp_dtensor"):
        raise ValueError("Home expert checkpoints require torch_dist or fsdp_dtensor format.")
    if checkpoint_args is not None:
        for name in (
            "moe_enable_scheduler",
            "moe_scheduler_planner_type",
            "moe_scheduler_home_update_interval",
            "moe_scheduler_num_idle_experts",
            "num_experts",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "world_size",
        ):
            if getattr(args, name, None) != getattr(checkpoint_args, name, None):
                raise ValueError(f"Home expert checkpoint requires unchanged {name}.")


@torch.no_grad()
def restore_home_expert_checkpoint_weights(optimizer):
    """Rebuild both MXFP8 orientations from restored optimizer master weights.

    A dequantized rowwise checkpoint cannot reconstruct the independent
    columnwise quantization used by backward. Stage all chained optimizer
    shards before gathering a shared DDP chunk, just as after a home exchange.
    FSDP restores its compute weights from master-weight checkpoint tensors.
    """
    leaves = [
        leaf
        for leaf in optimizer_leaves(optimizer)
        if getattr(getattr(leaf, "config", None), "reuse_grad_buf_for_mxfp8_param_ag", False)
        and not getattr(getattr(leaf, "ddp_config", None), "use_megatron_fsdp", False)
        and not getattr(leaf, "is_stub_optimizer", False)
    ]
    chunks = {id(chunk): chunk for leaf in leaves for chunk in leaf.model_chunks}
    for chunk in chunks.values():
        chunk.zero_grad_buffer()
    for leaf in leaves:
        leaf._copy_main_params_to_param_buffer()
    for chunk in chunks.values():
        chunk.start_param_sync(force_sync=True)


def prepare_home_expert_step(model, optimizer):
    """Drain old work and defer parameter gathers if any layer has a pending plan.

    Returns the schedulers that need a successful-step notification. This is
    intentionally training-loop glue, not a combined expert dispatch component.
    """
    schedulers = [
        module
        for chunk in model
        for module in chunk.modules()
        if isinstance(module, MoEScheduler) and module.home_expert_dispatch is not None
    ]
    if any(s.home_expert_dispatch.has_pending for s in schedulers):
        for scheduler in schedulers:
            scheduler.expert_dispatch.assert_idle()
            home = scheduler.home_expert_dispatch
            if home.state_adapter is None:
                scheduler.bind_optimizer(optimizer)
        # First version uses a device fence at rare migration boundaries. It
        # includes FSDP/replica reduction streams and previous gather consumers.
        torch.cuda.synchronize()
        for leaf in optimizer_leaves(optimizer):
            leaf._home_exchange_defer_param_sync = True
    return schedulers


@torch.no_grad()
def finish_home_expert_step(schedulers, optimizer, update_successful):
    """Migrate updated state and commit mappings before restarting gathers."""
    leaves = optimizer_leaves(optimizer)
    deferred = any(getattr(leaf, "_home_exchange_defer_param_sync", False) for leaf in leaves)
    if update_successful:
        for scheduler in schedulers:
            scheduler.step()
        if deferred:
            # All layers migrate before copying/casting model weights. Existing
            # DDP/FSDP/MXFP8 code owns every destination wrapper and scale layout.
            for leaf in leaves:
                if getattr(leaf, "is_stub_optimizer", False):
                    continue
                config = getattr(leaf, "config", None)
                if getattr(config, "reuse_grad_buf_for_mxfp8_param_ag", False):
                    if not config.overlap_param_gather:
                        leaf._copy_main_params_to_param_buffer()
                elif hasattr(leaf, "_copy_main_params_to_model_params"):
                    leaf._copy_main_params_to_model_params()
    for leaf in leaves:
        leaf._home_exchange_defer_param_sync = False
    if deferred and update_successful:
        chunks = {}
        for leaf in leaves:
            ddp_config = getattr(leaf, "ddp_config", None)
            if ddp_config is not None and (
                ddp_config.use_megatron_fsdp or not ddp_config.overlap_param_gather
            ):
                for chunk in leaf.model_chunks:
                    chunks[id(chunk)] = chunk
        for chunk in chunks.values():
            chunk.start_param_sync()
        if getattr(
            getattr(optimizer, "config", None), "overlap_param_gather_with_optimizer_step", False
        ):
            first_chunk = leaves[0].model_chunks[0]
            if id(first_chunk) not in chunks:
                first_chunk.start_param_sync(force_dispatch=True)
