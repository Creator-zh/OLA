$ErrorActionPreference = "Stop"

python -m experiments.deltanet_ola.mqar.train `
  --methods ola `
  --training-steps 2 `
  --training-batch-size 2 `
  --evaluation-batch-count 1 `
  --model-hidden-sizes 16 `
  --state-matrix-dimension 8 `
  --sequence-length 16 `
  --number-of-key-value-pairs 2 `
  --vocabulary-size 64 `
  --device cpu `
  --out-jsonl runs\deltanet_ola\smoke_mqar.jsonl `
  --out-csv runs\deltanet_ola\smoke_mqar.csv
