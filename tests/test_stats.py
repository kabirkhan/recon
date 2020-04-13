from recon.stats import (
    calculate_entity_coverage_entropy,
    calculate_entity_coverage_similarity,
    calculate_label_balance_entropy,
    calculate_label_distribution_similarity,
    detect_outliers,
    get_entity_coverage,
    get_ner_stats,
)
from recon.types import NERStats


def test_get_ner_stats(example_corpus_processed):
    stats: NERStats = get_ner_stats(example_corpus_processed.train)  # type: ignore
    str_stats = get_ner_stats(example_corpus_processed.train, serialize=True)
    assert isinstance(str_stats, str)

    assert stats.n_examples == 106
    assert stats.n_examples_no_entities == 29
    assert stats.n_annotations_per_type == {"SKILL": 199, "PRODUCT": 34, "JOB_ROLE": 10}


def test_calculate_label_distribution_similarity(example_corpus_processed):
    similarity = calculate_label_distribution_similarity(
        example_corpus_processed.train, example_corpus_processed.dev
    )

    assert round(similarity, 2) == 86.44


def test_calculate_entity_coverage_similarity(example_corpus_processed):
    ec_stats = calculate_entity_coverage_similarity(
        example_corpus_processed.train, example_corpus_processed.dev
    )

    assert round(ec_stats.entity, 2) == 31.21
    assert round(ec_stats.count, 2) == 36.67


def test_calculate_entity_coverage_entropy(example_corpus_processed):
    entity_coverage = get_entity_coverage(example_corpus_processed.train)
    entropy = calculate_entity_coverage_entropy(entity_coverage)

    assert round(entropy, 2) == 5.24


def test_calculate_label_balance_entropy(example_corpus_processed):
    ner_stats = get_ner_stats(example_corpus_processed.train)
    entropy = calculate_label_balance_entropy(ner_stats)  # type: ignore
    assert round(entropy, 2) == 0.57


def test_detect_outliers():
    seq = [-1, 10, 11, 12, 12, 13, 14, 15, 16, 16, 17, 99]

    outliers = detect_outliers(seq)
    assert outliers.low == [0]
    assert outliers.high == [11]
