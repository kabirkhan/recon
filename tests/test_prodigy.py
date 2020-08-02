import pytest
from recon.corpus import Corpus

prodigy = pytest.importorskip("prodigy")


def test_corpus_to_from_prodigy(example_corpus):
    prodigy_train_dataset, prodigy_dev_dataset, prodigy_test_dataset = example_corpus.to_prodigy(
        overwrite=True
    )
    name = example_corpus.name

    corpus_loaded = Corpus.from_prodigy(
        name, [prodigy_train_dataset], [prodigy_dev_dataset], [prodigy_test_dataset]
    )

    assert example_corpus.name == corpus_loaded.name
    assert len(example_corpus.train) == len(corpus_loaded.train)
    # assert len(example_corpus.dev) == len(corpus_loaded.dev)
    # assert len(example_corpus.test) == len(corpus_loaded.test)
    assert len(example_corpus.all) == len(corpus_loaded.all)
