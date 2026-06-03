# OLA MQAR Minimal Experiment

This standalone PyTorch experiment compares a DeltaNet-style recurrence against the proposed OLA Cayley-transition recurrence on a single-query associative recall sanity check.

This is not the Zoology / DeltaNet-paper MQAR benchmark. The high scores here come from a much easier task: each sample contains key-value pairs followed by exactly one query key at the final position, and the model predicts one value token.

## Run Tests

```powershell
python -m pytest tests/ola_mqar -q
```

## Smoke Train

```powershell
python -m experiments.ola_mqar.train --method delta --steps 20 --eval-interval 10 --batch-size 16 --eval-batches 2 --vocab-size 64 --num-pairs 2 --d-model 32 --state-dim 16
python -m experiments.ola_mqar.train --method ola --steps 20 --eval-interval 10 --batch-size 16 --eval-batches 2 --vocab-size 64 --num-pairs 2 --d-model 32 --state-dim 16
```

## Paired Comparison

```powershell
python -m experiments.ola_mqar.compare --steps 200 --eval-interval 100 --batch-size 32 --eval-batches 5 --num-pairs 2 --vocab-size 32 --d-model 32 --state-dim 16
```

## Sweep

Run a repeatable sweep and save both JSONL and CSV:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 2,4,8 --d-model-list 32,64 --seeds 1,2,3 --steps 500 --eval-interval 250 --batch-size 64 --eval-batches 20 --vocab-size 128 --state-dim 32 --quiet-train
```

Single-query small experiment:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 2,4,8 --d-model-list 128 --seeds 1,2,3 --steps 1000 --eval-interval 250 --batch-size 64 --eval-batches 50 --vocab-size 128 --state-dim 32 --lr 0.003 --quiet-train
```

CPU-friendly smoke sweep:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 2 --d-model-list 16 --seeds 1 --steps 20 --eval-interval 10 --batch-size 4 --eval-batches 2 --vocab-size 64 --state-dim 8 --quiet-train
```

Default outputs:

```text
runs/ola_mqar/sweep.jsonl
runs/ola_mqar/sweep.csv
```

## Larger First Comparison

```powershell
python -m experiments.ola_mqar.train --method delta --steps 2000 --eval-interval 200 --batch-size 128 --eval-batches 50 --vocab-size 128 --num-pairs 8 --d-model 128 --state-dim 64 --save runs/delta.pt
python -m experiments.ola_mqar.train --method ola --steps 2000 --eval-interval 200 --batch-size 128 --eval-batches 50 --vocab-size 128 --num-pairs 8 --d-model 128 --state-dim 64 --save runs/ola.pt
```

The key comparison fields are `eval_loss` and `eval_acc`, computed on the single final query target. OLA also reports `orthogonality_error`; values close to zero indicate that the Cayley transition is numerically orthogonal.

## Minimal Multi-Query Validator

This script is a separate sanity check for the multi-query label structure only. It does not train a model.

```powershell
python -m experiments.ola_mqar.validate_multiquery
```

## Paper-Style MQAR: FLA DeltaNet vs OLA

`paper_mqar_experiment.py` is the GPU-oriented entry point for matching the MQAR setting used by the DeltaNet paper more closely. It calls `fla.layers.DeltaNet` for the baseline and uses OLA for the proposed method.

Required in the GPU environment:

```powershell
pip install flash-linear-attention
```

Full paper-style default command:

```powershell
python -m experiments.ola_mqar.paper_mqar_experiment --device cuda
```

The defaults are:

```text
vocabulary_size = 8192
sequence_lengths = 512
number_of_key_value_pairs = 64
model_hidden_sizes = 64,128,256,512
random_seeds = 1,2,3
delta_num_heads = 2
delta_use_short_conv = false
```

For a single GPU trial before launching the full sweep:

```powershell
python -m experiments.ola_mqar.paper_mqar_experiment --device cuda --model-hidden-sizes 64 --random-seeds 1 --training-steps 1000 --training-batch-size 64 --evaluation-batch-count 8
```
