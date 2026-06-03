from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.ola_mqar.data import MQARConfig, generate_mqar_batch
from experiments.ola_mqar.model import MQARModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Delta or OLA on a small MQAR task.")
    parser.add_argument("--method", choices=["delta", "ola"], required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-data-seed", type=int, default=None)
    parser.add_argument("--eval-data-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_eval_batches(
    data_config: MQARConfig,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
):
    generator = make_generator(device, seed)
    return [
        generate_mqar_batch(
            data_config,
            batch_size=batch_size,
            device=device,
            generator=generator,
        )
        for _ in range(eval_batches)
    ]


@torch.no_grad()
def evaluate(
    model: MQARModel,
    eval_data,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    last_aux: dict[str, torch.Tensor] = {}
    for batch in eval_data:
        logits, aux = model(batch.input_ids)
        loss = F.cross_entropy(logits, batch.labels)
        batch_size = batch.input_ids.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == batch.labels).sum().item()
        total_count += batch_size
        last_aux = aux
    aux_floats = {key: float(value.item()) for key, value in last_aux.items()}
    return total_loss / total_count, total_correct / total_count, aux_floats


def train_method(
    method: str,
    steps: int,
    eval_interval: int,
    batch_size: int,
    eval_batches: int,
    vocab_size: int,
    num_pairs: int,
    d_model: int,
    state_dim: int,
    lr: float,
    seed: int,
    device: torch.device,
    train_data_seed: int | None = None,
    eval_data_seed: int | None = None,
    log: bool = True,
) -> tuple[MQARModel, dict[str, float]]:
    torch.manual_seed(seed)
    train_data_seed = seed + 100_000 if train_data_seed is None else train_data_seed
    eval_data_seed = seed + 200_000 if eval_data_seed is None else eval_data_seed
    data_config = MQARConfig(vocab_size=vocab_size, num_pairs=num_pairs)
    train_generator = make_generator(device, train_data_seed)
    eval_data = build_eval_batches(
        data_config=data_config,
        batch_size=batch_size,
        eval_batches=eval_batches,
        device=device,
        seed=eval_data_seed,
    )
    model = MQARModel(
        method=method,
        vocab_size=vocab_size,
        d_model=d_model,
        state_dim=state_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    start = time.perf_counter()
    metrics: dict[str, float] = {}
    for step in range(1, steps + 1):
        model.train()
        batch = generate_mqar_batch(
            data_config,
            batch_size=batch_size,
            device=device,
            generator=train_generator,
        )
        logits, _ = model(batch.input_ids)
        loss = F.cross_entropy(logits, batch.labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            elapsed = time.perf_counter() - start
            eval_loss, eval_acc, aux = evaluate(model=model, eval_data=eval_data)
            metrics = {
                "train_loss": loss.item(),
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "steps_per_sec": step / max(elapsed, 1e-9),
                "model_parameter_count": float(model_parameter_count),
                "model_initialization_seed": float(seed),
                "training_data_seed": float(train_data_seed),
                "evaluation_data_seed": float(eval_data_seed),
                **aux,
            }
            aux_text = " ".join(f"{key}={value:.3e}" for key, value in sorted(aux.items()))
            if log:
                print(
                    f"method={method} step={step} train_loss={loss.item():.4f} "
                    f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                    f"steps_per_sec={metrics['steps_per_sec']:.2f} {aux_text}",
                    flush=True,
                )

    return model, metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = train_method(
        method=args.method,
        steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        d_model=args.d_model,
        state_dim=args.state_dim,
        lr=args.lr,
        seed=args.seed,
        device=device,
        train_data_seed=args.train_data_seed,
        eval_data_seed=args.eval_data_seed,
        log=True,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "args": vars(args),
                "model": model.state_dict(),
            },
            args.save,
        )


if __name__ == "__main__":
    main()
