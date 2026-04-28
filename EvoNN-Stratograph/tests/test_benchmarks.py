from pathlib import Path

from stratograph.benchmarks import get_benchmark, list_benchmarks, load_parity_pack
from stratograph.benchmarks.spec import BenchmarkSpec


def test_builtin_benchmark_count() -> None:
    names = [spec.name for spec in list_benchmarks()]
    assert len(names) >= 38
    assert "tiny_lm_synthetic" in names
    assert "wikitext2_lm_smoke" in names


def test_synthetic_lm_loads() -> None:
    spec = get_benchmark("tiny_lm_synthetic")
    x_train, y_train, x_val, y_val = spec.load_data(seed=7)
    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_train.shape[1] == 128
    assert x_val.shape[1] == 128
    assert y_train.shape == x_train.shape
    assert y_val.shape == x_val.shape


def test_tier1_regression_benchmarks_load() -> None:
    for name in ("diabetes", "friedman1"):
        spec = get_benchmark(name)
        x_train, y_train, x_val, y_val = spec.load_data(seed=7)
        assert x_train.ndim == 2
        assert x_val.ndim == 2
        assert y_train.ndim == 1
        assert y_val.ndim == 1
        assert x_train.shape[0] > 0
        assert x_val.shape[0] > 0


def test_simple_pack_parses(repo_root) -> None:
    pack = load_parity_pack(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    assert pack.name == "working_33_plus_5_lm_smoke"
    assert len(pack.benchmarks) == 38
    assert pack.benchmarks[0].metric_name == "accuracy"


def test_local_csv_spec_loads(tmp_path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("f1,f2,target\n0,1,a\n1,0,b\n0,0,a\n1,1,b\n", encoding="utf-8")
    spec = BenchmarkSpec(
        name="toy_local",
        task="classification",
        source="local",
        path=str(Path(csv_path)),
        target_column="target",
        input_dim=2,
        num_classes=2,
    )
    x_train, y_train, x_val, y_val = spec.load_data(seed=42, validation_split=0.5)
    assert x_train.shape[1] == 2
    assert x_val.shape[1] == 2
    assert set(y_train.tolist() + y_val.tolist()) == {0, 1}
