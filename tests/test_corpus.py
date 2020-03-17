from pathlib import Path

from recon.corpus import Corpus
from recon.stats import get_ner_stats


def test_dataset_initialize(example_data):
    ds1 = Corpus(example_data["train"], example_data["dev"])
    assert len(ds1.train) == 102
    assert len(ds1.dev) == 110
    assert len(ds1.all) == 212

    ds2 = Corpus(example_data["train"], example_data["dev"], test=example_data["test"])
    assert len(ds2.train) == 102
    assert len(ds2.dev) == 110
    assert len(ds2.test) == 96
    assert len(ds2.all) == 308


def test_dataset_from_disk(example_data):
    ds = Corpus.from_disk(Path(__file__).parent.parent / "examples/data/skills")
    assert ds.train == example_data["train"]
    assert ds.dev == example_data["dev"]
    assert ds.test == example_data["test"]
    assert len(ds.all) == 308


def test_dataset_apply(example_data):
    dataset = Corpus(
        example_data["train"], example_data["dev"], test=example_data["test"]
    )
    stats = dataset.apply(get_ner_stats)
    assert sorted(list(stats.keys())) == ["all", "dev", "test", "train"]

    assert stats["all"].n_examples == 308
