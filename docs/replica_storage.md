# MoEScheduler replica storage

Peer TMA and NCCL P2P support two independent options:

| Option | Values | Default |
| --- | --- | --- |
| `--moe-scheduler-replica-storage` | `shared`, `layerwise` | `shared` |
| `--moe-scheduler-replica-mxfp8-storage` | `reuse`, `separate` | `reuse` |

`shared` reuses replica weights, scales, and gradients among compatible layers
on the same EP group/device. `layerwise` allocates private replica storage.
Native-gradient staging follows its separate runtime sharing contract.
`reuse` aliases rowwise and columnwise MXFP8 storage; `separate` retains both
orientations. The second option has no effect on BF16 allocation.

For example, enable directional caching with:

```sh
--moe-scheduler-replica-storage layerwise \
--moe-scheduler-replica-mxfp8-storage separate
```

The runtime lazily fills each directional cache. A hit skips replica transport,
not native expert/GTP materialization or completion waits. Cache validity uses
exact placement and local source-byte/scale snapshots, not plan object identity
or Tensor version counters. This detects optimizer updates, checkpoint writes,
requantization, and home migration even when writes bypass version counters.
An EP-wide miss reduction ensures all senders and receivers agree. Snapshot
storage is an additional per-layer cost in this opt-in mode; comparisons and
the control collective also have a runtime cost. BF16 layerwise storage supports
the same cache with its single plain layout.

Shared storage and MXFP8 reuse mode do not enable this cache optimization.
The first backward must populate columnwise data; a cached rowwise tensor is
not a substitute for independently quantized columnwise bytes. An older
microbatch's placement can require a refresh after another placement overwrote
its directional cache. No gradients are cached: each microbatch still computes
and reduces its own replica gradients.

## Consumer lifetime

A runtime acquires a group/device consumer fence before prefetch and releases
it after token combine in forward, or after gradient publication in backward.
The next producer waits on the last consumer event. Communication completion
alone does not authorize overwrite. Unreleased/reentrant consumers raise an
error instead of overwriting shared storage. Source producers and GTP's
materialize/consume protocol remain intact on cache hits.

NCCL workspaces share the communication stream and completed-transfer cleanup,
so layer-local lists do not retain old temporary messages until that layer is
visited again. Peer-TMA symmetric windows have explicit transport reference
counts; destroying one shared user does not invalidate the others.
Completed NCCL handles retain only their completion event and transport;
the workspace releases packed messages after their GPU work completes, even
when a layer keeps an old handle for repeated waits.

## CUDA graphs and diagnostics

The exact cache requires host decisions and cannot run inside a captured MoE
region. Attention-only CUDA graphs remain supported. NCCL retains its existing
restriction against captured host-scheduled MoE communication.
`runtime.weight_cache.hits` and `.misses` expose collective cache decisions for
tests and profiling. Source snapshots and cache entries are ephemeral and are
not checkpoint state.

## Tests

`tests/unit_tests/transformer/moe/test_replica_storage.py` exercises both backends
and all four policies with BF16/MXFP8, exact bytes/scales, additive gradients,
consumer streams, rank-asymmetric cache invalidation, old placement restoration,
and real TE MXFP8 forward/dgrad GEMMs with repeated runtime prefetch requests.
The existing NCCL transport and runtime boundary suites remain regression tests.
