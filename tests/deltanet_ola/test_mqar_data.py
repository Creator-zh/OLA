import torch


def test_paper_mqar_defaults_match_delta_net_figure_2():
    from experiments.deltanet_ola.mqar.data import MQARConfig

    config = MQARConfig()

    assert config.vocabulary_size == 8192
    assert config.sequence_length == 512
    assert config.number_of_key_value_pairs == 64
    assert config.power_a == 0.01
    assert config.random_non_queries is True


def test_mqar_batch_labels_only_query_positions():
    from experiments.deltanet_ola.mqar.data import MQARConfig, generate_mqar_batch

    config = MQARConfig(vocabulary_size=64, sequence_length=16, number_of_key_value_pairs=2)
    batch = generate_mqar_batch(config, batch_size=4, device=torch.device("cpu"), generator=torch.Generator().manual_seed(1))

    assert batch.input_ids.shape == (4, 16)
    assert batch.labels.shape == (4, 16)
    assert batch.query_mask.shape == (4, 16)
    assert torch.equal(batch.query_mask, batch.labels != -100)
    assert batch.query_mask.sum().item() == 8
