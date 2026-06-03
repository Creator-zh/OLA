from pathlib import Path


def test_short_lm_presets_exist_for_4090_workflow():
    root = Path("experiments/deltanet_ola/lm/configs")

    assert (root / "short_smoke.yaml").exists()
    assert (root / "short_10m.yaml").exists()
    assert (root / "short_100m.yaml").exists()
