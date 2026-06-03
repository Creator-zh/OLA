from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare short LM reproduction commands for DeltaNet and OLA.")
    parser.add_argument("--config", type=Path, default=Path("experiments/deltanet_ola/lm/configs/short_smoke.yaml"))
    parser.add_argument("--method", choices=["fla_delta", "ola"], default="fla_delta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    print(f"config={args.config}")
    print(f"method={args.method}")
    print("Use 3rdparty/flash-linear-attention/legacy/training as the aligned LM training stack.")
    print("Prepare the tokenized SlimPajama subset before launching distributed training.")


if __name__ == "__main__":
    main()

