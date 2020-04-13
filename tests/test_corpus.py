import shutil
from pathlib import Path

import pytest
from recon.corpus import Corpus
from recon.stats import get_ner_stats

# def test_corpus_initialize(example_data):
#     corpus1 = Corpus(example_data["train"], example_data["dev"])
#     assert len(corpus1.train) == 106
#     assert len(corpus1.dev) == 110
#     assert len(corpus1.all) == 216

#     corpus2 = Corpus(example_data["train"], example_data["dev"], test=example_data["test"])
#     assert len(corpus2.train) == 106
#     assert len(corpus2.dev) == 110
#     assert len(corpus2.test) == 96
#     assert len(corpus2.all) == 312


# def test_corpus_disk(example_data):
#     corpus = Corpus(example_data["train"], example_data["dev"], test=example_data["test"])
#     save_dir = Path(__file__).parent.parent / "examples/data/skills/to_disk_test"

#     corpus.to_disk(save_dir, force=True)
#     corpus2 = Corpus.from_disk(save_dir)
#     assert corpus.train == example_data["train"]
#     assert corpus.dev == example_data["dev"]
#     assert corpus.test == example_data["test"]

#     assert corpus2.train == example_data["train"]
#     assert corpus2.dev == example_data["dev"]
#     assert corpus2.test == example_data["test"]
#     assert len(corpus.all) == len(corpus2.all)

#     shutil.rmtree(save_dir)


# def test_corpus_apply(example_data):
#     corpus = Corpus(example_data["train"], example_data["dev"], test=example_data["test"])
#     stats = corpus.apply(get_ner_stats)
#     assert sorted(list(stats.keys())) == ["all", "dev", "test", "train"]

#     assert stats["all"].n_examples == 312


# def test_corpus_apply_inplace(example_data):
#     corpus = Corpus(example_data["train"], example_data["dev"], test=example_data["test"])
#     stats = corpus.apply(get_ner_stats)
#     assert sorted(list(stats.keys())) == ["all", "dev", "test", "train"]

#     assert stats["all"].n_examples == 312
