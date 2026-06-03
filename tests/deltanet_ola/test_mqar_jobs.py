def test_paper_sweep_jobs_cross_methods_hidden_sizes_and_seeds():
    from experiments.deltanet_ola.mqar.sweep import MQARSweepConfig, build_mqar_jobs

    config = MQARSweepConfig(
        methods=("fla_delta", "ola"),
        model_hidden_sizes=(64, 128),
        random_seeds=(1, 2),
    )

    jobs = build_mqar_jobs(config)

    assert len(jobs) == 8
    assert jobs[0]["method"] == "fla_delta"
    assert jobs[0]["model_hidden_size"] == 64
    assert jobs[0]["random_seed"] == 1
    assert jobs[-1]["method"] == "ola"
    assert jobs[-1]["model_hidden_size"] == 128
    assert jobs[-1]["random_seed"] == 2
