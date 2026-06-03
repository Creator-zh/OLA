import torch

from experiments.ola_mqar.data import MQARConfig, generate_mqar_batch


def test_generate_mqar_batch_returns_mqar_sequence_labels_and_query_positions():
    config = MQARConfig(vocab_size=128, input_seq_len=32, num_kv_pairs=4, random_non_queries=False)

    batch = generate_mqar_batch(config, batch_size=8, device=torch.device("cpu"))

    assert batch.input_ids.shape == (8, 32)
    assert batch.labels.shape == (8, 32)
    assert batch.query_mask.shape == (8, 32)
    assert batch.query_mask.sum().item() == 8 * config.num_kv_pairs
    assert torch.all(batch.labels[~batch.query_mask] == -100)

    key_positions = torch.arange(0, config.num_kv_pairs * 2, 2)
    value_positions = torch.arange(1, config.num_kv_pairs * 2, 2)
    for b in range(batch.input_ids.shape[0]):
        mapping = {
            int(batch.input_ids[b, key_pos]): int(batch.input_ids[b, value_pos])
            for key_pos, value_pos in zip(key_positions, value_positions)
        }
        for position in torch.nonzero(batch.query_mask[b], as_tuple=False).flatten():
            query_token = int(batch.input_ids[b, position])
            assert int(batch.labels[b, position]) == mapping[query_token]


def test_generate_mqar_batch_is_reproducible_with_explicit_generator():
    config = MQARConfig(vocab_size=128, input_seq_len=32, num_kv_pairs=4)
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
