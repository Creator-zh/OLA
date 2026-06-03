from experiments.ola_mqar.validate_multiquery import build_multiquery_example, validate_multiquery_example


def test_validate_multiquery_example_checks_all_query_labels():
    example = build_multiquery_example(
        num_key_value_pairs=3,
        filler_token=0,
    )

    assert validate_multiquery_example(example)
    assert sum(label != -100 for label in example.labels) == 3
