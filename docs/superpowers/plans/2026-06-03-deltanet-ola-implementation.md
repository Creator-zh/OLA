# DeltaNet OLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify this project so the main experiment path reproduces the DeltaNet paper MQAR and short language-model benchmark settings, with OLA compared under the same wrappers.

**Architecture:** Keep the existing OLA layer implementation reusable, add `fla-org/flash-linear-attention` as a pinned submodule, and create a new paper-aligned package under `experiments/deltanet_ola`. The existing `experiments/ola_mqar` minimal experiment is retained only as a legacy sanity check and no longer presented as the main workflow.

**Tech Stack:** Python, PyTorch, pytest, FLA DeltaNet from `3rdparty/flash-linear-attention`, optional CUDA for paper-sized runs.

---

### Task 1: Source Alignment

**Files:**
- Create: `.gitmodules`
- Create: `3rdparty/flash-linear-attention`
- Create: `experiments/deltanet_ola/fla_paths.py`
- Test: `tests/deltanet_ola/test_fla_paths.py`

- [ ] Add `fla-org/flash-linear-attention` as a Git submodule at `3rdparty/flash-linear-attention`.
- [ ] Add a helper that prepends the submodule root to `sys.path` when present.
- [ ] Test that the helper returns the expected path and is idempotent.

### Task 2: Shared Metadata and Metrics

**Files:**
- Create: `experiments/deltanet_ola/env.py`
- Create: `experiments/deltanet_ola/metrics.py`
- Test: `tests/deltanet_ola/test_metrics.py`

- [ ] Implement run metadata collection for git commit, FLA commit, PyTorch version, CUDA availability, GPU names, dtype, and device count.
- [ ] Implement a benchmark timer context and peak CUDA memory reader that also works on CPU.
- [ ] Test metadata shape and CPU-safe memory reporting.

### Task 3: Method Adapters

**Files:**
- Create: `experiments/deltanet_ola/methods.py`
- Modify: `experiments/ola_mqar/layers.py` only if the reusable OLA output contract needs a small compatibility fix.
- Test: `tests/deltanet_ola/test_methods.py`

- [ ] Implement method names `fla_delta` and `ola`.
- [ ] FLA DeltaNet adapter must import `fla.layers.DeltaNet` through the submodule helper.
- [ ] OLA adapter must reuse `experiments.ola_mqar.layers.OLAMixer`.
- [ ] Test that the OLA adapter produces sequence-shaped hidden states on CPU.
- [ ] Test that the FLA adapter raises a clear dependency message when FLA cannot be imported.

### Task 4: Paper-Aligned MQAR

**Files:**
- Create: `experiments/deltanet_ola/mqar/data.py`
- Create: `experiments/deltanet_ola/mqar/train.py`
- Create: `experiments/deltanet_ola/mqar/sweep.py`
- Create: `experiments/deltanet_ola/mqar/benchmark.py`
- Test: `tests/deltanet_ola/test_mqar_data.py`
- Test: `tests/deltanet_ola/test_mqar_jobs.py`

- [ ] Implement MQAR config defaults from the DeltaNet paper: sequence length 512, 64 key-value pairs, vocabulary size 8192, hidden sizes 64/128/256/512, seeds 1/2/3, 2 DeltaNet heads, no short convolution.
- [ ] Generate labels only at query positions, with ignored labels elsewhere.
- [ ] Implement training and evaluation loops that produce query loss, query accuracy, tokens/sec, steps/sec, peak memory, parameter count, and metadata.
- [ ] Implement paper sweep job generation across methods, hidden sizes, and seeds.
- [ ] Implement benchmark presets for smoke and paper-sized runs.

### Task 5: Short LM Scaffold

**Files:**
- Create: `experiments/deltanet_ola/lm/configs/short_smoke.yaml`
- Create: `experiments/deltanet_ola/lm/configs/short_10m.yaml`
- Create: `experiments/deltanet_ola/lm/configs/short_100m.yaml`
- Create: `experiments/deltanet_ola/lm/train_short.py`
- Create: `experiments/deltanet_ola/lm/benchmark.py`
- Test: `tests/deltanet_ola/test_lm_configs.py`

- [ ] Add LM config presets sized for 4 x RTX 4090.
- [ ] Add a launcher that records exact commands and explains required external datasets/tokenizers.
- [ ] Add benchmark-only mode for sequence-length throughput checks.
- [ ] Keep full 340M long training disabled by default.

### Task 6: Mainline Docs and Legacy Demotion

**Files:**
- Create: `experiments/deltanet_ola/README.md`
- Modify: `experiments/ola_mqar/README.md`
- Modify: root docs only if needed.

- [ ] Document `experiments/deltanet_ola` as the main experiment path.
- [ ] Mark `experiments/ola_mqar` as legacy minimal sanity code.
- [ ] Remove outdated wording that suggests the old minimal experiment is the paper reproduction.

### Task 7: Verification

**Commands:**

```powershell
python -m pytest tests/deltanet_ola tests/ola_mqar -q
python -m experiments.deltanet_ola.mqar.train --methods ola --training-steps 2 --training-batch-size 2 --evaluation-batch-count 1 --model-hidden-sizes 16 --state-matrix-dimension 8 --sequence-length 16 --number-of-key-value-pairs 2 --vocabulary-size 64 --device cpu
```

- [ ] Tests pass locally.
- [ ] CPU smoke run writes a row with throughput and metadata.
- [ ] If CUDA and FLA are installed, run a one-step `fla_delta` smoke command.
- [ ] Commit all changes.
