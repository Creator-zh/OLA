from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.ola_mqar.data import MQARConfig
from experiments.ola_mqar.model import MQARModel
from experiments.ola_mqar.train import build_eval_batches, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved MQAR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-data-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    train_args = payload["args"]

    data_config = MQARConfig(
        vocab_size=train_args["vocab_size"],
        input_seq_len=train_args["input_seq_len"],
        num_kv_pairs=train_args["num_pairs"],
        power_a=train_args["power_a"],
    )
    model = MQARModel(
        method=train_args["method"],
        vocab_size=train_args["vocab_size"],
        d_model=train_args["d_model"],
        state_dim=train_args["state_dim"],
    ).to(device)
    model.load_state_dict(payload["model"])

    eval_data = build_eval_batches(
        data_config=data_config,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        device=device,
        seed=args.eval_data_seed,
    )
    loss, acc, aux = evaluate(model, eval_data)
    aux_text = " ".join(f"{key}={value:.3e}" for key, value in sorted(aux.items()))
    print(f"method={train_args['method']} eval_loss={loss:.4f} eval_acc={acc:.4f} {aux_text}")


if __name__ == "__main__":
    main()
