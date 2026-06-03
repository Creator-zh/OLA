# DeltaNet Paper Reproduction and OLA Comparison

This is the main experiment path for reproducing the DeltaNet paper settings and comparing OLA on the same workloads.

The DeltaNet baseline is sourced from the pinned FLA submodule:

```text
3rdparty/flash-linear-attention
```

## MQAR

Default MQAR settings follow the DeltaNet paper Figure 2:

```text
sequence_length = 512
number_of_key_value_pairs = 64
vocabulary_size = 8192
model_hidden_sizes = 64,128,256,512
random_seeds = 1,2,3
delta_num_heads = 2
delta_use_short_conv = false
```

CPU OLA smoke:

```powershell
python -m experiments.deltanet_ola.mqar.train --methods ola --training-steps 2 --training-batch-size 2 --evaluation-batch-count 1 --model-hidden-sizes 16 --state-matrix-dimension 8 --sequence-length 16 --number-of-key-value-pairs 2 --vocabulary-size 64 --device cpu
```

Paper-sized run on CUDA after installing FLA dependencies:

```powershell
python -m experiments.deltanet_ola.mqar.train --device cuda
```

Results are written to:

```text
runs/deltanet_ola/mqar.jsonl
runs/deltanet_ola/mqar.csv
```

## Short LM

Short LM presets for 4 x RTX 4090 live under:

```text
experiments/deltanet_ola/lm/configs
```

These preserve the paper-style optimizer and data assumptions while keeping the first training budgets practical for local validation.

The full 340M long-training configuration should be added after MQAR and short LM smoke runs validate OLA stability and throughput.
