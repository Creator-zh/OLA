# OLA MQAR Minimal Experiment

This standalone PyTorch experiment compares a DeltaNet-style recurrence against the proposed OLA Cayley-transition recurrence on Zoology-style Multi-Query Associative Recall (MQAR).

## Run Tests

```powershell
python -m pytest tests/ola_mqar -q
```

## Smoke Train

```powershell
python -m experiments.ola_mqar.train --method delta --steps 20 --eval-interval 10 --batch-size 8 --eval-batches 2 --vocab-size 128 --input-seq-len 32 --num-pairs 4 --d-model 32 --state-dim 16
python -m experiments.ola_mqar.train --method ola --steps 20 --eval-interval 10 --batch-size 8 --eval-batches 2 --vocab-size 128 --input-seq-len 32 --num-pairs 4 --d-model 32 --state-dim 16
```

## Paired Comparison

```powershell
python -m experiments.ola_mqar.compare --steps 100 --eval-interval 50 --batch-size 16 --eval-batches 5 --vocab-size 128 --input-seq-len 32 --num-pairs 4 --d-model 32 --state-dim 16
```

## Sweep

Run a repeatable sweep and save both JSONL and CSV:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 4 --d-model-list 32,64 --seeds 1,2,3 --steps 300 --eval-interval 150 --batch-size 32 --eval-batches 20 --vocab-size 1024 --input-seq-len 64 --state-dim 32 --quiet-train
```

CPU Figure-4-style small experiment:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 4 --d-model-list 32,64,128 --seeds 1,2,3 --steps 500 --eval-interval 250 --batch-size 32 --eval-batches 50 --vocab-size 1024 --input-seq-len 64 --state-dim 32 --lr 0.003 --quiet-train
```

CPU-friendly smoke sweep:

```powershell
python -m experiments.ola_mqar.run_sweep --num-pairs-list 4 --d-model-list 32 --seeds 1 --steps 20 --eval-interval 10 --batch-size 8 --eval-batches 2 --vocab-size 128 --input-seq-len 32 --state-dim 16 --quiet-train
```

Default outputs:

```text
runs/ola_mqar/sweep.jsonl
runs/ola_mqar/sweep.csv
```

## Larger First Comparison

```powershell
python -m experiments.ola_mqar.train --method delta --steps 2000 --eval-interval 200 --batch-size 128 --eval-batches 50 --vocab-size 8192 --input-seq-len 512 --num-pairs 64 --d-model 128 --state-dim 64 --save runs/delta.pt
python -m experiments.ola_mqar.train --method ola --steps 2000 --eval-interval 200 --batch-size 128 --eval-batches 50 --vocab-size 8192 --input-seq-len 512 --num-pairs 64 --d-model 128 --state-dim 64 --save runs/ola.pt
```

The key comparison fields are `eval_loss` and `eval_acc`, computed only on MQAR query positions. OLA also reports `orthogonality_error`; values close to zero indicate that the Cayley transition is numerically orthogonal.
