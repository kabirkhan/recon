from pathlib import Path

from recon.dataset import Dataset
from recon.stats import ner_stats


def test_dataset_initialize(example_data):
    ds1 = Dataset(example_data["train"], example_data["dev"])
    assert len(ds1.train) == 102
    assert len(ds1.dev) == 110
    assert len(ds1.all) == 212

    ds2 = Dataset(example_data["train"], example_data["dev"], test=example_data["test"])
    assert len(ds2.train) == 102
    assert len(ds2.dev) == 110
    assert len(ds2.test) == 96
    assert len(ds2.all) == 308


def test_dataset_from_disk(example_data):
    ds = Dataset.from_disk(Path(__file__).parent.parent / "examples/data")
    assert ds.train == example_data["train"]
    assert ds.dev == example_data["dev"]
    assert ds.test == example_data["test"]
    assert len(ds.all) == 308


def test_dataset_apply(example_data):
    dataset = Dataset(
        example_data["train"], example_data["dev"], test=example_data["test"]
    )
    stats = dataset.apply(ner_stats)
    assert sorted(list(stats.keys())) == ["all", "dev", "test", "train"]

    assert stats["all"]["n_examples"] == 308
