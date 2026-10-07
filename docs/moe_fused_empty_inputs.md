# Empty inputs in fused experts

A rank can receive no tokens under an imbalanced MoE route, including when the
scheduler is disabled. TE activation kernels must not receive a zero-row TMA
descriptor. The fused expert path pads an empty input to one aligned group of
zero inputs and zero probabilities, executes normal parameter hooks and expert
operations, then returns the empty output slice. Input and probability gradients
retain their original empty shapes; expert gradients remain zero and participate
in the ordinary DDP/optimizer lifecycle.

This is separate from scheduler placement or dispatch capacity. Nonempty inputs
retain the existing path. `test_fused_expert_empty_rank.py` exercises real TE,
HybridEP, DDP and distributed Adam with all routed work concentrated on one of
four ranks and checks zero expert gradients on the other ranks.
