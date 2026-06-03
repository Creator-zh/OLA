from experiments.ola_mqar.paper_mqar_experiment import PaperMQARConfig, build_paper_jobs


def test_paper_mqar_config_uses_delta_paper_defaults():
    config = PaperMQARConfig()

    assert config.vocabulary_size == 8192
    assert config.sequence_lengths == (512,)
    assert config.number_of_key_value_pairs == (64,)
    assert config.model_hidden_sizes == (64, 128, 256, 512)
    assert config.delta_num_heads == 2
    assert config.delta_use_short_conv is False


def test_build_paper_jobs_crosses_methods_dimensions_and_seeds():
    config = PaperMQARConfig(
        methods=("fla_delta", "ola"),
        model_hidden_sizes=(64, 128),
        random_seeds=(1, 2),
        sequence_lengths=(512,),
        number_of_key_value_pairs=(64,),
    )

    jobs = build_paper_jobs(config)

    assert len(jobs) == 8
    assert jobs[0]["method"] == "fla_delta"
    assert jobs[0]["model_hidden_size"] == 64
    assert jobs[0]["random_seed"] == 1
    assert jobs[-1]["method"] == "ola"
    assert jobs[-1]["model_hidden_size"] == 128
    assert jobs[-1]["random_seed"] == 2
