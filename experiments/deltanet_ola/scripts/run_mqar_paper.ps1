$ErrorActionPreference = "Stop"

python -m experiments.deltanet_ola.mqar.train `
  --methods fla_delta,ola `
  --device cuda `
  --out-jsonl runs\deltanet_ola\mqar_paper.jsonl `
  --out-csv runs\deltanet_ola\mqar_paper.csv
