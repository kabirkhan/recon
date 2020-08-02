import shutil
from pathlib import Path

from recon.corpus import Corpus
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def test_corpus_initialize(example_data):

    train_ds = Dataset("train")
    dev_ds = Dataset("dev")

    corpus_base = Corpus("test_corpus", train_ds, dev_ds)
    assert corpus_base.name == "test_corpus"

    assert len(corpus_base.train) == 0
    assert len(corpus_base.dev) == 0
    assert len(corpus_base.test) == 0
    assert len(corpus_base.all) == 0

    train_ds = Dataset("train", example_data["train"])
    dev_ds = Dataset("dev", example_data["dev"])
    test_ds = Dataset("test", example_data["test"])

    corpus1 = Corpus("corpus1", train_ds, dev_ds)
    assert len(corpus1.train) == 106
    assert len(corpus1.dev) == 110
    assert len(corpus1.all) == 216

    corpus_loaded = Corpus("corpus_loaded", train_ds, dev_ds, test_ds)
    assert len(corpus_loaded.train) == 106
    assert len(corpus_loaded.dev) == 110
    assert len(corpus_loaded.test) == 96
    assert len(corpus_loaded.all) == 312


def test_corpus_disk(example_data):
    train_ds = Dataset("train", example_data["train"])
    dev_ds = Dataset("dev", example_data["dev"])
    test_ds = Dataset("test", example_data["test"])
    corpus = Corpus("test_corpus", train_ds, dev_ds, test_ds)

    save_dir = Path(__file__).parent.parent / "examples/data/skills/to_disk_test"

    corpus.to_disk(save_dir, force=True)
    corpus_loaded = Corpus.from_disk(save_dir)
    assert corpus.train == example_data["train"]
    assert corpus.dev == example_data["dev"]
    assert corpus.test == example_data["test"]

    assert corpus_loaded.train == example_data["train"]
    assert corpus_loaded.dev == example_data["dev"]
    assert corpus_loaded.test == example_data["test"]
    assert len(corpus.all) == len(corpus_loaded.all)

    assert corpus_loaded.name == corpus.name

    shutil.rmtree(save_dir)


def test_corpus_apply(example_corpus):
    stats = example_corpus.apply(get_ner_stats)
    assert stats.all.n_examples == 312


def test_corpus_apply_inplace(example_corpus_processed):
    assert len(example_corpus_processed.train_ds.operations) == 4
    assert len(example_corpus_processed.dev_ds.operations) == 4
    assert len(example_corpus_processed.test_ds.operations) == 4
