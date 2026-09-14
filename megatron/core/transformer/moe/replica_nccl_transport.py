# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reserved portable NCCL point-to-point backend; no communication yet."""

from megatron.core.transformer.moe.replica_weight_transport import (
    UnimplementedReplicaWeightTransport,
)


class NcclP2PTransport(UnimplementedReplicaWeightTransport):
    """Future packed peer transfers with local unpack and gradient accumulation.

    Compile matching peer schedules from placement and ownership, documenting
    any host synchronization. Completion must include unpack/FP32 accumulation.
    Initialize communicators consistently and preserve buffer lifetimes across
    NCCL's internal streams. Dynamic host plans must not claim graph support.
    """

    transport_name = "replica_nccl"
