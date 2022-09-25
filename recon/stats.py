import math
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Union, cast

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy

from recon.constants import NOT_LABELED
from recon.types import EntityCoverage, EntityCoverageStats, Example, Outliers, Stats


def get_ner_stats(data: List[Example], return_examples: bool = False) -> Stats:
    """Compute statistics for NER data

    Args:
        data (Iterator[Example]): Data as a List of examples
        return_examples (bool, optional): Whether to return examples per type

    Returns:
        Stats: Summary stats from list of Examples
    """
    annotations_per_type: DefaultDict[str, Any] = defaultdict(int)
    examples: DefaultDict[str, Any] = defaultdict(list)
    n_examples_no_entities = 0

    for e in data:
        if not e.spans:
            n_examples_no_entities += 1
            examples[NOT_LABELED].append(e)
        else:
            for s in e.spans:
                annotations_per_type[s.label] += 1
                examples[s.label].append(e)

    sorted_anns_by_count = {
        a[0]: a[1] for a in sorted(annotations_per_type.items(), key=lambda x: x[1], reverse=True)
    }

    stats = Stats(
        n_examples=len(data),
        n_examples_no_entities=n_examples_no_entities,
        n_annotations=sum(annotations_per_type.values()),
        n_annotations_per_type=sorted_anns_by_count,
    )
    if return_examples:
        stats.examples_with_type = examples

    return stats


def get_sorted_type_counts(ner_stats: Stats) -> List[int]:
    """Get list of counts for each type in n_annotations_per_type property
    of an Stats object sorted by type name

    Args:
        ner_stats (Stats): Dataset stats

    Returns:
        List[int]: List of counts sorted by type name
    """
    annotations_per_type = ner_stats.n_annotations_per_type
    annotations_per_type[NOT_LABELED] = ner_stats.n_examples_no_entities
    return [t[1] for t in sorted(annotations_per_type.items(), key=lambda p: p[0])]


def calculate_label_distribution_similarity(x: List[Example], y: List[Example]) -> float:
    """Calculate the similarity of the label distribution for 2 datasets.

    e.g. This can help you understand how well your train set models your dev and test sets.
    Empircally you want a similarity over **0.8** when comparing your train set to each of your
    dev and test sets.

        calculate_label_distribution_similarity(corpus.train, corpus.dev)
        # 98.57

        calculate_label_distribution_similarity(corpus.train, corpus.test)
        # 73.29 - This is bad, let's investigate our test set more

    Args:
        x (List[Example]): Dataset
        y (List[Example]): Dataset to compare x to

    Returns:
        float: Similarity of label distributions
    """

    def pipeline(data: List[Example]) -> Sequence[float]:
        stats = cast(Stats, get_ner_stats(data))
        sorted_type_counts = get_sorted_type_counts(stats)
        counts_to_probs = get_probs_from_counts(sorted_type_counts)
        return counts_to_probs

    distance = jensenshannon(pipeline(x), pipeline(y))
    return float(1 - distance) * 100


def get_entity_coverage(
    data: List[Example],
    sep: str = "||",
    case_sensitive: bool = False,
    return_examples: bool = False,
) -> List[EntityCoverage]:
    """Identify how well you dataset covers an entity type. Get insights
    on the how many times certain text/label span combinations exist across your
    data so that you can focus your annotation efforts better rather than
    annotating examples your Model already understands well.

    Args:
        data (List[Example]): List of examples
        sep (str, optional): Separator used in coverage map, only change if || exists in your text
            or label.
        case_sensitive (bool, optional): Consider case of text for each annotation
        return_examples (bool, optional): Return Examples that contain the entity label annotation.

    Returns:
        List[EntityCoverage]: Sorted List of EntityCoverage objects containing the text, label, count, and
            an optional list of examples where that text/label annotation exists.
    """
    coverage_map: DefaultDict[str, int] = defaultdict(int)
    examples_map: DefaultDict[str, List[Example]] = defaultdict(list)

    for example in data:
        for span in example.spans:
            text = span.text if case_sensitive else span.text.lower()
            key = f"{text}{sep}{span.label}"
            coverage_map[key] += 1
            examples_map[key].append(example)

    coverage = []
    for key, count in coverage_map.items():
        text, label = key.split(sep)
        record = EntityCoverage(text=text, label=label, count=count)
        if return_examples:
            record.examples = examples_map[key]
        coverage.append(record)

    sorted_coverage = sorted(coverage, key=lambda x: x.count, reverse=True)
    return sorted_coverage


def calculate_entity_coverage_similarity(x: List[Example], y: List[Example]) -> EntityCoverageStats:
    """Calculate how well dataset x covers the entities in dataset y.
    This function should be used to calculate how similar your train set
    annotations cover the annotations in your dev/test set

    Args:
        x (List[Example]): Dataset to compare coverage to (usually corpus.train)
        y (List[Example]): Dataset to evaluate coverage for (usually corpus.dev or corpus.test)

    Returns:
        EntityCoverageStats: Stats with
            1. The base entity coverage (does entity in y exist in x)
            2. Count coverage (sum of the EntityCoverage.count property for
            each EntityCoverage in y to get a more holisic coverage scaled by how
            often entities occur in each dataset x and y)
    """

    def pipeline(data: List[Example]) -> Dict[int, int]:
        ecs = get_entity_coverage(data)
        return {hash(ec): ec.count for ec in ecs}

    x_map = pipeline(x)
    y_map = pipeline(y)

    n_intersection = 0
    count_intersection = 0
    n_union = 0
    count_union = 0

    for k, count in y_map.items():
        if k in x_map:
            n_intersection += 1
            count_intersection += count
        n_union += 1
        count_union += count

    return EntityCoverageStats(
        entity=(n_intersection / n_union) * 100,
        count=(count_intersection / count_union) * 100,
    )


def get_probs_from_counts(seq: List[int]) -> Sequence[float]:
    """Convert a sequence of counts to a sequence of probabilties
    by dividing each n by the sum of all n in seq

    Args:
        seq (Sequence[int]): Sequence of counts

    Returns:
        Sequence[float]: Sequence of probabilities
    """
    return list(np.asarray(seq) / sum(seq))


def _entropy(seq: Union[List[int], List[float]], total: Optional[int] = None) -> float:
    """Calculate Shannon Entropy for a sequence of Floats or Integers.
    If Floats, check they are probabilities
    If Integers, divide each n in seq by total and calculate entropy

    Args:
        seq (Union[List[int], List[float]]): Sequence to calculate entropy for
        total (int, optional): Total to divide by for List of int

    Raises:
        ValueError: If seq is not valid

    Returns:
        float: Entropy for sequence
    """
    if not seq:
        raise ValueError("Pass a valid non-empty sequence")

    if isinstance(seq[0], float):
        res = scipy_entropy(seq)
    elif isinstance(seq[0], int):
        seq = cast(List[int], seq)
        res = scipy_entropy(get_probs_from_counts(seq))
    else:
        raise ValueError("Parameter seq must be a sequence of probabilites or integers.")
    return float(res)


def calculate_label_balance_entropy(ner_stats: Stats) -> float:
    """Use Entropy to calculate a metric for label balance based on a Stats object

    Args:
        ner_stats (Stats): Stats for a dataset.

    Returns:
        float: Entropy for annotation counts of each label
    """
    total = ner_stats.n_annotations
    classes = [count for _, count in ner_stats.n_annotations_per_type.items()]
    return _entropy(classes, total)


def calculate_entity_coverage_entropy(
    entity_coverage: List[EntityCoverage],
) -> float:
    """Use Entropy to calculate a metric for entity coverage.

    Args:
        entity_coverage (List[EntityCoverage]): List of EntityCoverage
            from get_entity_coverage

    Returns:
        float: Entropy for entity coverage counts
    """
    counts = [ecs.count for ecs in entity_coverage]
    return _entropy(counts, sum(counts))  # type: ignore


def detect_outliers(seq: Sequence[Any], use_log: bool = False) -> Outliers:
    """Detect outliers in a numerical sequence.

    Args:
        seq (Sequence[Any]): Sequence of ints or floats
        use_log (bool, optional): Use logarithm of seq.

    Returns:
        Tuple[List[int], List[int]]: Tuple of low and high indices
    """
    q1 = np.quantile(seq, 0.25)
    q3 = np.quantile(seq, 0.75)
    iqr = q3 - q1
    fence_low = math.floor(q1 - 1.5 * iqr)
    fence_high = math.floor(q3 + 1.5 * iqr)
    low_indices = [i for i, n in enumerate(seq) if n <= fence_low]
    high_indices = [i for i, n in enumerate(seq) if n > fence_high]
    return Outliers(low=low_indices, high=high_indices)
