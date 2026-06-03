from pathlib import Path


def test_fla_root_points_to_submodule():
    from experiments.deltanet_ola.fla_paths import fla_root

    root = fla_root()

    assert root == Path("3rdparty/flash-linear-attention").resolve()
    assert (root / "fla" / "layers" / "delta_net.py").exists()


def test_ensure_fla_on_path_is_idempotent():
    from experiments.deltanet_ola.fla_paths import ensure_fla_on_path, fla_root

    before = ensure_fla_on_path()
    after = ensure_fla_on_path()

    assert before == fla_root()
    assert after == before
