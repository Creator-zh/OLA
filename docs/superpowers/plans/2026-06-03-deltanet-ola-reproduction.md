# DeltaNet Paper Reproduction and OLA Comparison Plan

## Goal

Reproduce the DeltaNet experiments from `arXiv:2406.06484v5` in this project and compare the proposed OLA method against the official FLA DeltaNet implementation.

The work is split into two stages:

1. Short-run reproduction and benchmark on 4 x RTX 4090 24 GB.
2. 340M long-training configuration after OLA is validated.

The immediate implementation target is stage 1. Stage 2 should be prepared as configuration and launch scripts after stage 1 passes correctness and stability checks.

## Source Alignment

Use `fla-org/flash-linear-attention` as a Git submodule:

```text
3rdparty/flash-linear-attention
```

This keeps the DeltaNet baseline tied to a specific upstream commit. Experiment outputs must record:

- FLA submodule commit
- local project commit
- PyTorch, CUDA, Triton, and flash-linear-attention versions
- GPU model and GPU count
- dtype and distributed strategy

Do not copy DeltaNet source into this project unless a specific patch is required and documented.

## Paper Targets

Primary paper reference:

```text
Parallelizing Linear Transformers with the Delta Rule over Sequence Length
arXiv:2406.06484v5
```

Stage 1 must align with these parts first:

- Figure 2: MQAR synthetic benchmark.
- Figure 4: training throughput benchmark structure.
- Table 2 style language modeling setup, but with a short token budget.

Stage 2 prepares the 340M Table 2 long-training setup.

## Stage 1 Scope

### MQAR

Use the paper MQAR setting:

- sequence length: 512
- number of key-value pairs: 64
- vocabulary size: 8192 unless the upstream FLA/Zoology setting specifies otherwise
- model hidden sizes: 64, 128, 256, 512
- random seeds: 1, 2, 3
- DeltaNet heads: 2
- DeltaNet short convolution: disabled

Methods:

- `fla_delta`: official FLA DeltaNet layer or model path
- `ola`: OLA layer adapted to the same wrapper

Metrics:

- query accuracy
- query loss
- training steps per second
- tokens per second
- peak CUDA memory
- parameter count
- wall-clock runtime

### Short Language Modeling

Use a FLA/FLAME-compatible training path where possible. The short-run LM experiment should preserve the shape of the paper setup but use a practical token budget for 4 x RTX 4090.

Initial short-run target:

- tokenizer: Mistral tokenizer if available through the FLA training stack
- dataset: SlimPajama subset or the closest FLA-supported paper-compatible subset
- model scale: small debug config first, then a compact DeltaNet/OLA config sized for 4 x 4090
- token budget: configurable, with presets such as smoke, 10M, and 100M tokens
- dtype: bf16 where stable
- distributed: torchrun / DDP or FSDP according to the FLA training entry point

Metrics:

- train loss
- validation perplexity
- tokens per second
- samples per second
- step time mean and p50/p95
- peak CUDA memory per GPU
- optimizer state memory if available
- gradient overflow or NaN count

### Throughput Benchmark

Match the paper Figure 4 structure where feasible:

```text
2K x 8
4K x 4
8K x 2
16K x 1
```

Interpret these as sequence length x per-device or effective batch setting after checking the upstream benchmark script. For 4 x RTX 4090, allow smaller fallback presets if a setting does not fit.

Benchmark methods:

- FLA DeltaNet
- OLA

Record:

- tokens/sec
- step time
- peak memory
- max fitting batch size
- dtype
- activation checkpointing setting
- gradient accumulation setting

## Stage 2 Scope

Prepare but do not run by default:

- 340M DeltaNet paper-style long-training config
- 340M OLA config with matched parameter budget where possible
- 15B token schedule from the paper
- AdamW optimizer
- peak learning rate: 3e-4
- initial/final learning rate: 3e-5
- cosine schedule
- warmup: 0.5B tokens for 340M
- weight decay: 0.01
- gradient clipping: 1.0
- DeltaNet head dimension: 128
- convolution kernel size: 4 for convolution-enabled language model runs

Because the available hardware is 4 x RTX 4090 24 GB rather than the paper's 8 x H100 setup, the 340M config must include:

- gradient accumulation
- activation checkpointing option
- reduced microbatch presets
- resume-safe checkpointing
- estimated tokens/day logging

## Proposed Project Layout

```text
3rdparty/
  flash-linear-attention/

experiments/
  deltanet_ola/
    README.md
    env.py
    fla_paths.py
    methods.py
    metrics.py
    mqar/
      data.py
      train.py
      sweep.py
      benchmark.py
    lm/
      configs/
      train_short.py
      benchmark.py
      eval_lm_harness.py
    scripts/
      run_mqar_smoke.ps1
      run_mqar_paper.ps1
      run_lm_smoke.ps1
      run_benchmark.ps1

runs/
  deltanet_ola/
```

Existing `experiments/ola_mqar` code should remain as the minimal standalone sanity experiment. New paper-aligned work should live under `experiments/deltanet_ola`.

## Implementation Strategy

1. Add FLA as a submodule and record its commit.
2. Build a thin import/path helper so local scripts can import the submodule without modifying global Python state manually.
3. Implement method adapters:
   - official FLA DeltaNet baseline
   - OLA wrapper with the same input/output contract
4. Port or align the MQAR data/training path to the paper/FLA setting.
5. Add benchmark utilities for time, throughput, memory, and run metadata.
6. Add short LM training and benchmark entry points.
7. Add 340M long-training configs after short LM and MQAR pass.

## Verification

Required checks before claiming stage 1 complete:

- Unit tests for MQAR data shape and label placement.
- Unit tests for method adapter output shape.
- OLA orthogonality check remains bounded on a sampled transition.
- CPU smoke test for MQAR with tiny dimensions.
- CUDA smoke test for MQAR if FLA is installed and GPU is available.
- Benchmark output writes JSONL and CSV with required metadata.
- Short LM smoke test initializes both DeltaNet and OLA methods.

## Open Decisions

These should be resolved during implementation after inspecting the FLA submodule:

- Exact upstream FLA entry point for language training.
- Whether FLA has a maintained MQAR/Zoology script or whether this project should keep its own paper-aligned generator.
- Exact SlimPajama subset path and tokenizer acquisition method.
- Whether OLA should first use a pure PyTorch recurrent implementation for correctness or needs a chunkwise/parallel implementation before LM benchmarking.
- Whether parameter matching should be exact or only hidden-size matched for OLA vs DeltaNet.

## Non-Goals For Stage 1

- Full 1.3B / 100B token reproduction.
- Full 3B / 1T token reproduction.
- Reproducing every baseline from the paper.
- Rewriting FLA kernels.
- Claiming paper-level LM quality from short token-budget runs.
