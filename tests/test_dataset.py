from pathlib import Path
from typing import cast

import pytest
from recon.corrections import corrections_from_dict
from recon.dataset import Dataset
from recon.stats import get_ner_stats
from recon.store import ExampleStore
from recon.types import Correction, NERStats, OperationStatus, TransformationType


def test_dataset_initialize(example_data):

    dataset = Dataset("train")
    assert dataset.name == "train"
    assert dataset.data == []
    assert dataset.example_store._map == {}
    assert dataset.commit_hash == "94efdd6f628eda9c1ae893467c9652808443ef3e"
    assert dataset.operations == []

    store = ExampleStore()
    dataset2 = Dataset("dev", example_data["dev"], [], store)
    assert dataset2.name == "dev"
    assert dataset2.data == example_data["dev"]
    assert dataset2.example_store == store
    assert dataset2.commit_hash == "dd05e54668c166d075bc4406bfee590e4c89a292"
    assert dataset2.operations == []


def test_dataset_commit_hash(example_data):
    train_dataset = Dataset("train", example_data["train"][:-1])
    dev_dataset = Dataset("train", example_data["dev"])

    assert train_dataset.commit_hash != dev_dataset.commit_hash

    train_commit = train_dataset.commit_hash
    train_dataset.data.append(example_data["train"][-1])

    assert train_dataset.commit_hash != train_commit
    assert hash(train_dataset) == 1186038092183970443


def test_len(example_data):
    train_dataset = Dataset("train", example_data["train"])
    assert len(train_dataset) == len(example_data["train"])


def test_apply(example_data):
    train_dataset = Dataset("train", example_data["train"])

    ner_stats: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))
    ner_stats_apply: NERStats = cast(NERStats, get_ner_stats(train_dataset.data))

    assert ner_stats.n_examples == ner_stats_apply.n_examples
    assert ner_stats.n_examples_no_entities == ner_stats_apply.n_examples_no_entities
    assert ner_stats.n_annotations == ner_stats_apply.n_annotations
    assert ner_stats.n_annotations_per_type == ner_stats_apply.n_annotations_per_type


def test_apply_(example_data):
    train_dataset = Dataset("train", example_data["train"])
    ner_stats_pre: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    assert len(train_dataset.operations) == 0

    train_dataset.apply_("recon.v1.upcase_labels")

    ner_stats_post: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    pre_keys = sorted(ner_stats_pre.n_annotations_per_type.keys())
    post_keys = sorted(ner_stats_post.n_annotations_per_type.keys())

    assert pre_keys != post_keys

    assert pre_keys == ["JOB_ROLE", "PRODUCT", "SKILL", "product", "skill"]
    assert post_keys == ["JOB_ROLE", "PRODUCT", "SKILL"]

    assert len(train_dataset.operations) == 1

    op = train_dataset.operations[0]

    assert op.name == "recon.v1.upcase_labels"
    assert op.status == OperationStatus.COMPLETED
    assert len(op.transformations) == 3

    for t in op.transformations:
        assert t.type == TransformationType.EXAMPLE_CHANGED


def test_rollback(example_data):
    train_dataset = Dataset("train", example_data["train"])
    ner_stats_pre: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    assert len(train_dataset.operations) == 0

    train_dataset.apply_("recon.v1.upcase_labels")

    ner_stats_post: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    pre_keys = sorted(ner_stats_pre.n_annotations_per_type.keys())
    post_keys = sorted(ner_stats_post.n_annotations_per_type.keys())

    assert pre_keys != post_keys

    assert pre_keys == ["JOB_ROLE", "PRODUCT", "SKILL", "product", "skill"]
    assert post_keys == ["JOB_ROLE", "PRODUCT", "SKILL"]

    assert len(train_dataset.operations) == 1

    train_dataset.rollback()

    assert len(train_dataset.operations) == 0

    ner_stats_rolled_back: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))
    rolled_back_keys = sorted(ner_stats_rolled_back.n_annotations_per_type.keys())

    assert pre_keys == rolled_back_keys


def test_dataset_search(example_data):
    train_dataset = Dataset("train", example_data["train"])

    assert len(train_dataset.search("kotlin")) == 0
    assert len(train_dataset.search("Kotlin")) == 1
    assert len(train_dataset.search("kotlin", case_sensitive=False)) == 1
    assert len(train_dataset.search("Software")) == 2
    assert len(train_dataset.search("Software", case_sensitive=False)) == 4


def test_dataset_to_from_disk(example_data, tmp_path):

    train_dataset = Dataset("train", example_data["train"])
    ner_stats_pre: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    assert len(train_dataset.operations) == 0

    with pytest.raises(FileNotFoundError):
        train_dataset.to_disk(tmp_path)

    train_dataset.to_disk(tmp_path, force=True)
    train_dataset_loaded = Dataset("train").from_disk(tmp_path)
    assert len(train_dataset_loaded.operations) == 0
    assert train_dataset_loaded.commit_hash == train_dataset.commit_hash

    train_dataset.apply_("recon.v1.upcase_labels")
    corrections = corrections_from_dict(
        {"software development engineer": "JOB_ROLE", "model": None}
    )
    assert len(corrections) == 2
    train_dataset.apply_("recon.v1.fix_annotations", corrections)

    train_dataset.to_disk(tmp_path, force=True)
    train_dataset_loaded_2 = Dataset("train").from_disk(tmp_path)

    assert len(train_dataset_loaded_2.operations) == 2
    assert train_dataset_loaded_2.commit_hash == train_dataset.commit_hash
    assert train_dataset_loaded_2.commit_hash != train_dataset_loaded.commit_hash

    op = train_dataset_loaded_2.operations[0]

    assert op.name == "recon.v1.upcase_labels"
    assert op.status == OperationStatus.COMPLETED
    assert len(op.transformations) == 3

    for t in op.transformations:
        assert t.type == TransformationType.EXAMPLE_CHANGED

    op2 = train_dataset_loaded_2.operations[1]

    assert op2.kwargs["corrections"] == [
        Correction(
            annotation="software development engineer", from_labels=["ANY"], to_label="JOB_ROLE"
        ).dict(),
        Correction(annotation="model", from_labels=["ANY"], to_label=None).dict(),
    ]


def test_dataset_to_from_spacy(example_data, tmp_path):
    train_dataset = Dataset("train", example_data["train"])
    ner_stats_pre: NERStats = cast(NERStats, train_dataset.apply(get_ner_stats))

    train_dataset.to_spacy(tmp_path)
    train_dataset_loaded = Dataset("train").from_spacy(Path(tmp_path) / "train.spacy")

    ner_stats_post: NERStats = cast(NERStats, train_dataset_loaded.apply(get_ner_stats))
    assert ner_stats_pre == ner_stats_post
