$ErrorActionPreference = "Stop"

python -m experiments.deltanet_ola.lm.train_short `
  --config experiments\deltanet_ola\lm\configs\short_smoke.yaml `
  --method ola
