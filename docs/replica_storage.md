# MoEScheduler replica storage

## Initialization and routing contract

Every enabled scheduler invocation executes `update_placement`, dispatch, and
`reroute`; there is no planning-skip hook. ECHO uses `one_shot_greedy` only.
The removed approximate bin-packing value is rejected during configuration.
Unimplemented HybridEP weight transport, MoonEP/peer-TMA EP sizes above 64,
and MoonEP expert counts above 8192 are rejected before allocation. MoonEP
requires equal physical token counts on all EP ranks; post-planner variable
token equalization cannot satisfy this requirement and is rejected.

Learned and mock routing retain exactly top-k assignments on padding rows
with zero probabilities. Padding contributes physical work to the planner but
no expert output or gate gradient. Scheduler-disabled dropless HybridEP keeps
its existing sparse padding behavior.

CUDA, explicit NCCL EP groups, HybridEP, Triton, TE grouped ops, and the fused
GroupedMLP environment setting are checked before constructing expert weights.
MoonEP's symmetric histogram window is initialized at that boundary. Home
optimizer representations and physical parameter bindings are checked during
model/optimizer initialization, not at the first migration. Supported home
optimizers are Adam and SGD with resident, uncompressed state.

## Checkpoint transitions

Loading a scheduler-disabled checkpoint with home migration enabled starts a
new training run: weights are loaded in identity home order, optimizer and
training/RNG progress are not restored, and counters start at zero. The source
must contain topology metadata and use the same EP/TP/PP/CP/world topology.
No exchange occurs during loading. The first forward initializes planner
history and normal `update_placement` proposals trigger later migrations at
successful optimizer boundaries. No planner state is required only for this
explicitly detected logical-weight initialization. Missing planner state in a
home checkpoint remains fatal, even with non-strict loading.

Resuming a home checkpoint requires unchanged topology and scheduler settings.
Changing parallel topology is unsupported. To disable the scheduler, first
export a canonical logical checkpoint using the original topology and scheduler
configuration:

```sh
# Append to the original model/training command; conversion saves and exits.
--load /path/to/home-checkpoint \
--ckpt-convert-format torch_dist \
--ckpt-convert-save /path/to/logical-export \
--moe-scheduler-export-logical-checkpoint
```

For Megatron-FSDP, use `--ckpt-convert-format fsdp_dtensor` instead; the output
is an ordinary logical FSDP checkpoint under the `fsdp_dtensor` subdirectory.
Export preserves the optimizer representation and saves synchronously before exit.

Export requires optimizer masters/state (`--finetune`, `--no-load-optim`, and
`--no-save-optim` are incompatible). It restores committed physical homes to
logical expert order together with master weights and optimizer tensors, then
regenerates compute weights and saves with scheduler metadata disabled. A
pending, uncommitted planner proposal is not applied. The output is an ordinary
checkpoint under `/path/to/logical-export/torch_dist`, loadable with the scheduler
disabled, including optimizer continuation. Export is terminal: the in-memory
token dispatcher is not reconfigured for continued execution in that process.

Peer TMA and NCCL P2P support two independent options:

| Option | Values | Default |
| --- | --- | --- |
| `--moe-scheduler-replica-storage` | `shared`, `layerwise` | `shared` |
| `--moe-scheduler-replica-mxfp8-storage` | `reuse`, `separate` | `reuse` |

`shared` reuses replica weights, scales, and gradients among compatible layers
on the same EP group/device. `layerwise` allocates private replica storage.
Regular DDP native gradients alias persistent `main_grad` buffers. FSDP, GTP,
and overwrite callers retain the separate native-gradient staging contract.
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

An unset `--moe-expert-rank-capacity-factor` preserves dropless dispatch for both
transports. ECHO/EPLB use HybridEP's dynamic output sizing. MoonEP supplies a
per-invocation bound on padded rows; HybridEP consumes it without adding padding
a second time and checks overflow on-device, avoiding dynamic output sizing in
eager execution. The scheduler does not imply a static
capacity of 1.0: a planner can leave a rank above the mean token count, and
per-expert padding can increase the required capacity further. Explicit static
capacity remains supported and requires the training overflow retry path when
the budget is exceeded. Capturing MoE requires an explicit static capacity;
the eager MoonEP bound does not relax that configuration restriction.

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

`test_scheduler_bf16_layer_parity.py` compares a real scheduler-disabled
MoELayer against each transport, using identical initial weights, inputs and
upstream gradients. On four GB200 GPUs it checks forward, input gradients,
router/expert gradients, and one distributed Adam update with one or four
microbatches. It checks FP32 master-weight deltas as well as BF16 weights, so
the initial parameter magnitude cannot hide an incorrect optimizer update.
This is a paired production-path diagnostic, not an independent mathematical
reference. A separate insufficient-static-capacity control must report both
dispatch overflow and numerical disagreement. The six cases pass per rank;
dynamic-capacity forward and input gradients match exactly in this recipe.


## Module boundaries

`MoEScheduler` owns the training-facing prepare, step, logical-load, and export
operations. Planners provide logical export placements through `MoELoadPlanner`;
training code never interprets committed or pending planner layouts. Home dispatch
has its own typed contract because it stages `HomeExpertPlacement`, unlike replica
dispatch's Tensor placement. It owns installation health and terminal export.
Replica dispatch binds the borrowed transport and optimizer adapter without
exposing its runtime to scheduler orchestration.

Optimizer leaves expose `get_parameter_state_binding`,
`defer_param_sync_for_state_update`, and `refresh_model_weights`. Bucket maps,
master-to-compute casts, and FSDP shard offsets remain owned by optimizer/FSDP.
The home adapter receives source parameters and transport configuration, not the
whole replica runtime. Expert modules own runtime detachment and fused-op cleanup.
Scheduler diagnostics use module loggers at INFO level, following application
logging configuration; they do not print directly to stdout.


The 2026-09-29 boundary refactor passed the release-contract suite on four
Lyris GB200 GPUs: 42 passed and one EP8 case skipped per rank. This includes
learned padding/autograd, distributed MoonEP routing, logical checkpoint load
and export with Adam continuation, failed-install protection, module logging,
optimizer hold release, parameter bindings, and configuration capacity checks.
The run used the approved ARM experimental TE image, not the official CI image.
The complete home-exchange, BF16 parity, and DataLoader RNG files were not rerun
in this validation; full FSDP/MXFP8 training remains outside this result.


## PR #7450 alignment (no scheduler-interface or GTP migration)

The MoonEP kernels track PR #7450 at `242c8dbc63c17ec804f1e31974d3dec86e5ed56c`.
The scheduler still calls `update_placement`, replica dispatch, then `reroute`.
Placement and reroute remain separate launches and retain the dense routing IR.
Consequently this branch does not claim the PR's fused planner/side-stream
scheduling overhead or end-to-end training performance.

Placement rounds each global logical expert count once into GEMM-aligned blocks
(256 for the supported fused grouped MLP), balances these blocks, and assigns
remainder blocks to lower ranks. Routes retain their source-rank/token order;
conceptual padding occurs only at the logical expert's end. Real route totals
can differ between destinations. This applies to BF16 as well as MXFP8.
The 32-program planner handles up to two EP ranks per program. Reroute uses
512-route sort/scan tiles and binary search while emitting the existing dense
probabilities with autograd. Compiled launchers are cached by shape, dtype,
alignment and input-pointer alignment. Token shape can change between eager
invocations; each shape gets its own specialization and capacity.

For R routes per source rank, P ranks, E logical experts and alignment A,
`blocks = (P*R + min(E, P*R)*(A-1)) // A` and the padded capacity is
`A * ceil(blocks/P)`. The optional padded-capacity metadata is independent of
an unpadded route bound. Explicit user capacity and overflow-retry policy take
precedence over both bounds. The existing explicit-capacity requirement for
captured MoE remains in place.

Peer-TMA pushes use the PR's TMA reads and coalesced uint64 peer stores for both
weight bytes and quantization scales. Existing transport completion and consumer
fences remain in force. Regular DDP natives accumulate directly into persistent
`main_grad`; virtual slots are zeroed before each accumulating backward. The
handoff triggers original parameter hooks without re-adding native gradients.
Runtime leaves do not request dummy gradients from TE. Overwrite/FSDP/GTP callers
retain staging; GTP gather/reduction behavior is unchanged. Native staging is
still allocated by the transport for fallback compatibility.

An unbound runtime can be destroyed without changing the expert layer's native
fused ops. Detaching a runtime owned by another layer remains an error.

The existing TE weighted squared-ReLU expert path is allowed by scheduler
configuration, with `ScaledSReLU` checked during environment initialization.
Quantile routing and upstream grouped-tensor/GTP API migration are outside this
change. All comparisons use a common supported recipe, not the PR's GTP recipe.

`test_moonep_pr7450_alignment.py` checks the pinned PR's independent CPU oracle,
local EP33/64 layouts, real EP communication and graph replay.
`test_moonep_pr7450_training.py` checks DDP accumulation and BF16/MXFP8 Adam
updates with mock/learned routing and weighted squared-ReLU. Local EP64 simulation is
not a 64-GPU collective test. The optional `test_moonep_pr7450_benchmark.py`
requires the unmodified, hash-pinned PR kernel source and records both kernel-only
and equivalent dense-output timings; it does not enforce noisy CI speed gates.
