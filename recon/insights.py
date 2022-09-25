from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import numpy as np
from spacy.scorer import PRFScore
from wasabi import Printer

from recon.constants import NOT_LABELED
from recon.recognizer import EntityRecognizer
from recon.types import (
    Example,
    HardestExample,
    LabelDisparity,
    PredictionError,
    PredictionErrorExamplePair,
)


def get_ents_by_label(
    data: List[Example], case_sensitive: bool = False
) -> DefaultDict[str, List[str]]:
    """Get a dictionary of unique text spans by label for your data

    # TODO: Ok so this needs to return more than just a set for each label.

    We want to return a dictionary that maps labels to AnnotationCount objects where each
    AnnotationCount contains the text of the annotation text, the total number of times it's mentioned (e.g. what entity_coverage does)
    but also the examples it is in.

    So maybe I can get this info from entity_coverage? IDK but this is dumb rn and not very flexible.

    Maybe I should keep this function returning a set of strings for each label for compatability but I need the other way too
    so I know what to focus on in editing and a

    Args:
        data (List[Example]): List of examples
        case_sensitive (bool, optional): Consider case of text for each annotation

    Returns:
        DefaultDict[str, List[str]]: DefaultDict mapping label to sorted list of the unique
            spans annotated for that label.
    """
    annotations: DefaultDict[str, Set[str]] = defaultdict(set)
    sorted_annotations: DefaultDict[str, List[str]] = defaultdict(list)

    for e in data:
        for s in e.spans:
            span_text = s.text if case_sensitive else s.text.lower()
            annotations[s.label].add(span_text)

    for label, anns in annotations.items():
        sorted_annotations[label] = sorted(anns)

    return sorted_annotations


def get_label_disparities(
    data: List[Example], label1: str, label2: str, case_sensitive: bool = False
) -> Set[str]:
    """Identify annotated spans that have different labels in different examples

    Args:
        data (List[Example]): Input List of examples
        label1 (str): First label to compare
        label2 (str): Second label to compare
        case_sensitive (bool, optional): Consider case of text for each annotation

    Returns:
        Set[str]: Set of all unique text spans that overlap between label1 and label2
    """
    annotations = get_ents_by_label(data, case_sensitive=case_sensitive)
    return set(annotations[label1]).intersection(set(annotations[label2]))


def top_label_disparities(
    data: List[Example], case_sensitive: bool = False, dedupe: bool = False
) -> List[LabelDisparity]:
    """Identify annotated spans that have different labels
    in different examples for all label pairs in data.

    Args:
        data (List[Example]): Input List of examples
        case_sensitive (bool, optional): Consider case of text for each annotation
        dedupe (bool, optional): Whether to deduplicate for table view vs confusion matrix.
            False by default for easy confusion matrix display.

    Returns:
        List[LabelDisparity]: List of LabelDisparity objects for each label pair combination
            sorted by the number of disparities between them.
    """
    annotations = get_ents_by_label(data, case_sensitive=case_sensitive)
    label_disparities = {}
    for label1 in annotations.keys():
        for label2 in annotations.keys():
            if label1 != label2:
                intersection = set(annotations[label1]).intersection(set(annotations[label2]))
                n_disparities = len(intersection)
                if n_disparities > 0:
                    if dedupe:
                        input_hash = "||".join(sorted([label1, label2]))
                    else:
                        input_hash = "||".join([label1, label2])

                    label_disparities[input_hash] = LabelDisparity(
                        label1=label1, label2=label2, count=n_disparities
                    )

    return sorted(label_disparities.values(), key=lambda ld: ld.count, reverse=True)


def top_prediction_errors(
    recognizer: EntityRecognizer,
    data: List[Example],
    labels: List[str] = [],
    exclude_fp: bool = False,
    exclude_fn: bool = False,
    verbose: bool = False,
    return_examples: bool = False,
) -> List[PredictionError]:
    """Get a sorted list of examples your model is worst at predicting.

    Args:
        recognizer (EntityRecognizer): An instance of EntityRecognizer
        data (List[Example]): List of annotated Examples
        labels (List[str]): List of labels to get errors for.
            Defaults to the labels property of `recognizer`.
        exclude_fp (bool, optional): Flag to exclude False Positive errors.
        exclude_fn (bool, optional): Flag to exclude False Negative errors.
        verbose (bool, optional): Show verbose output.
        return_examples (bool, optional): Return Examples that contain the entity label annotation.

    Returns:
        List[PredictionError]: List of Prediction Errors your model is making, sorted by the
            spans your model has the most trouble with.
    """
    labels = labels or recognizer.labels
    texts = (e.text for e in data)
    anns = (e.spans for e in data)
    preds = recognizer.predict(texts)

    errors = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # type: ignore
    error_examples: DefaultDict[
        Tuple[str, str, str], List[PredictionErrorExamplePair]
    ] = defaultdict(list)
    n_errors = 0

    for orig_example, pred_example, ann in zip(data, preds, anns):

        pred_error_example_pair = PredictionErrorExamplePair(
            original=orig_example, predicted=pred_example
        )

        cand = set([(s.start, s.end, s.label) for s in pred_example.spans])
        gold = set([(s.start, s.end, s.label) for s in ann])

        fp_diff = cand - gold
        fn_diff = gold - cand

        seen = set()

        if fp_diff and not exclude_fp:
            for fp in fp_diff:
                gold_ent = None
                for ge in gold:
                    if fp[0] == ge[0] and fp[1] == ge[1]:
                        gold_ent = ge
                        break
                if gold_ent:
                    start, end, label = gold_ent
                    text = pred_example.text[start:end]
                    false_label = fp[2]
                    errors[label][text][false_label] += 1
                    error_examples[(text, label, false_label)].append(pred_error_example_pair)
                else:
                    start, end, false_label = fp
                    text = pred_example.text[start:end]
                    errors[NOT_LABELED][text][false_label] += 1
                    error_examples[(text, NOT_LABELED, false_label)].append(pred_error_example_pair)
                n_errors += 1
                seen.add((start, end))

        if fn_diff and not exclude_fn:
            for fn in fn_diff:
                start, end, label = fn
                if (start, end) not in seen:
                    text = pred_example.text[start:end]
                    errors[label][text][NOT_LABELED] += 1
                    error_examples[(text, label, NOT_LABELED)].append(pred_error_example_pair)
                    n_errors += 1

    ranked_errors_map: Dict[Tuple[str, str, str], PredictionError] = {}

    for label, errors_per_label in errors.items():
        for error_text, error_labels in errors_per_label.items():
            for error_label, count in error_labels.items():
                pe_hash = (error_text, label, error_label)
                pe = PredictionError(
                    text=error_text,
                    true_label=label,
                    pred_label=error_label,
                    count=count,
                )
                if return_examples:
                    pe.examples = error_examples[pe_hash]
                ranked_errors_map[pe_hash] = pe

    ranked_errors: List[PredictionError] = sorted(
        list(ranked_errors_map.values()), key=lambda error: error.count, reverse=True  # type: ignore
    )
    error_texts = set()
    for re in ranked_errors:
        if re.examples:
            for e in re.examples:
                error_texts.add(e.original.text)

    error_rate = round(len(error_texts) / len(data), 2)
    if verbose:
        error_summary = {
            "N Examples": len(data),
            "N Errors": len(ranked_errors),
            "N Error Examples": len(error_texts),
            "Error Rate": error_rate,
        }
        msg = Printer()
        msg.divider("Error Analysis")
        msg.table(error_summary)

    return ranked_errors


def get_hardest_examples(
    recognizer: EntityRecognizer,
    examples: List[Example],
    score_count: bool = True,
    normalize_scores: bool = True,
) -> List[HardestExample]:
    """Get hardest examples for a recognizer to predict on and sort by difficulty with the goal
    of quickly identifying the biggest holes in a model / annotated data.

    Args:
        recognizer (EntityRecognizer): EntityRecognizer to test predictions for
        examples (List[Example]): Set of input examples
        score_count (bool): Adjust score by total number of errors
        normalize_scores (bool): Scale scores to range [0, 1] adjusted by total number of errors

    Returns:
        List[HardestExample]: HardestExamples sorted by difficulty (hardest first)
    """
    preds = recognizer.predict((e.text for e in examples))

    max_count = 0
    hes = []
    for pred, ref in zip(preds, examples):

        scorer = PRFScore()
        scorer.score_set(
            set([(s.start, s.end, s.label) for s in pred.spans]),
            set([(s.start, s.end, s.label) for s in ref.spans]),
        )
        total_errors = scorer.fp + scorer.fn
        score = scorer.fscore if (pred.spans and ref.spans) else 1.0

        if total_errors > max_count:
            max_count = total_errors

        he = HardestExample(reference=ref, prediction=pred, count=total_errors, score=score)
        hes.append(he)

    if score_count:
        for he in hes:
            he.score -= he.count / max_count
        if normalize_scores:
            scores = np.asarray([he.score for he in hes])
            scores = (scores - scores.min()) / np.ptp(scores)
            for i, he in enumerate(hes):
                he.score = scores[i]

    sorted_hes = sorted(hes, key=lambda he: (he.score, he.count))
    return sorted_hes


def get_annotation_labels(
    examples: List[Example], case_sensitive: bool = False
) -> Dict[str, Dict[str, list]]:
    """Constructs a map of each annotation in the list of examples to each label that annotation
    has and references all examples associated with that label.

    Args:
        examples (List[Example]): Input examples
        case_sensitive (bool, optional): Consider case of text for each annotation

    Returns:
        Dict[str, Dict[str, list]]: Annotation map
    """
    annotation_labels_map: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in examples:
        for s in e.spans:
            text = s.text if case_sensitive else s.text.lower()
            annotation_labels_map[text][s.label].append(e)

    return annotation_labels_map
