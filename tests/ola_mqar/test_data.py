import torch

from experiments.ola_mqar.data import MQARConfig, generate_mqar_batch


def test_generate_mqar_batch_returns_expected_shapes_and_valid_labels():
    config = MQARConfig(vocab_size=128, num_pairs=4)

    batch = generate_mqar_batch(config, batch_size=8, device=torch.device("cpu"))

    assert batch.input_ids.shape == (8, 10)
    assert batch.labels.shape == (8,)
    assert batch.query_positions.shape == (8,)
    assert batch.value_positions.shape == (8,)
    assert torch.all(batch.input_ids[:, -2] == config.query_token_id)
    assert torch.all(batch.query_positions == 9)

    gathered_values = batch.input_ids[
        torch.arange(batch.input_ids.shape[0]),
        batch.value_positions,
    ]
    assert torch.equal(batch.labels, gathered_values)


def test_generate_mqar_batch_is_reproducible_with_explicit_generator():
    config = MQARConfig(vocab_size=128, num_pairs=4)
    generator_a = torch.Generator(device="cpu").manual_seed(123)
    generator_b = torch.Generator(device="cpu").manual_seed(123)

    batch_a = generate_mqar_batch(
        config,
        batch_size=8,
        device=torch.device("cpu"),
        generator=generator_a,
    )
    batch_b = generate_mqar_batch(
        config,
        batch_size=8,
        device=torch.device("cpu"),
        generator=generator_b,
    )

    assert torch.equal(batch_a.input_ids, batch_b.input_ids)
    assert torch.equal(batch_a.labels, batch_b.labels)
