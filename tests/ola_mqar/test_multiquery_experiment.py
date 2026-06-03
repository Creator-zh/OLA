import torch

from experiments.ola_mqar.multiquery_experiment import (
    MultiQueryConfig,
    MultiQueryModel,
    generate_multiquery_batch,
    query_accuracy,
)


def test_generate_multiquery_batch_contains_multiple_query_targets():
    config = MultiQueryConfig(
        vocabulary_size=64,
        sequence_length=16,
        number_of_key_value_pairs=2,
        random_non_queries=False,
    )

    batch = generate_multiquery_batch(config, batch_size=4, device=torch.device("cpu"))

    assert batch.input_ids.shape == (4, 16)
    assert batch.labels.shape == (4, 16)
    assert batch.query_mask.sum().item() == 8
    assert torch.all(batch.labels[~batch.query_mask] == -100)

    context_length = config.number_of_key_value_pairs * 2
    for batch_index in range(batch.input_ids.shape[0]):
        mapping = {
            int(batch.input_ids[batch_index, position]): int(batch.input_ids[batch_index, position + 1])
            for position in range(0, context_length, 2)
        }
        query_positions = torch.nonzero(batch.query_mask[batch_index], as_tuple=False).flatten()
        assert torch.all((query_positions - context_length) % 2 == 0)
        for query_position in query_positions:
            query_token = int(batch.input_ids[batch_index, query_position])
            assert int(batch.labels[batch_index, query_position]) == mapping[query_token]


def test_multiquery_model_scores_each_sequence_position():
    config = MultiQueryConfig(vocabulary_size=64, sequence_length=16, number_of_key_value_pairs=2)
    batch = generate_multiquery_batch(config, batch_size=4, device=torch.device("cpu"))
    model = MultiQueryModel(method="ola", vocabulary_size=64, model_hidden_size=16, state_matrix_dimension=8)

    logits, aux = model(batch.input_ids)

    assert logits.shape == (4, 16, 64)
    assert isinstance(aux, dict)
    assert 0.0 <= query_accuracy(logits, batch.labels, batch.query_mask) <= 1.0
