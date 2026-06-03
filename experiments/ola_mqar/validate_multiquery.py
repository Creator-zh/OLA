from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiQueryExample:
    input_ids: list[int]
    labels: list[int]
    key_value_count: int


def build_multiquery_example(
    num_key_value_pairs: int = 3,
    filler_token: int = 0,
) -> MultiQueryExample:
    keys = list(range(1, num_key_value_pairs + 1))
    values = list(range(101, 101 + num_key_value_pairs))
    input_ids: list[int] = []
    labels: list[int] = []

    for key, value in zip(keys, values):
        input_ids.extend([key, value])
        labels.extend([-100, -100])

    for key, value in zip(reversed(keys), reversed(values)):
        input_ids.extend([filler_token, key])
        labels.extend([-100, value])

    return MultiQueryExample(
        input_ids=input_ids,
        labels=labels,
        key_value_count=num_key_value_pairs,
    )


def validate_multiquery_example(example: MultiQueryExample) -> bool:
    context = example.input_ids[: example.key_value_count * 2]
    mapping = {
        context[index]: context[index + 1]
        for index in range(0, len(context), 2)
    }
    for token, label in zip(example.input_ids, example.labels):
        if label == -100:
            continue
        if mapping.get(token) != label:
            return False
    return True


def main() -> None:
    example = build_multiquery_example()
    print("input_ids:", example.input_ids)
    print("labels:   ", example.labels)
    print("valid:    ", validate_multiquery_example(example))


if __name__ == "__main__":
    main()
