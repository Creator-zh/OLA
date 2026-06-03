import csv
import json

from experiments.ola_mqar.results import append_jsonl, write_csv


def test_append_jsonl_and_write_csv_persist_metrics(tmp_path):
    rows = [
        {"method": "delta", "num_pairs": 2, "seed": 1, "eval_loss": 1.0, "eval_acc": 0.5},
        {"method": "ola", "num_pairs": 2, "seed": 1, "eval_loss": 0.9, "eval_acc": 0.6},
    ]

    jsonl_path = tmp_path / "results.jsonl"
    csv_path = tmp_path / "results.csv"

    for row in rows:
        append_jsonl(jsonl_path, row)
    write_csv(csv_path, rows)

    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert jsonl_rows == rows

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["method"] == "delta"
    assert csv_rows[1]["eval_acc"] == "0.6"
