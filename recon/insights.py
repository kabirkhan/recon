import copy
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Generator, List, Set, Union

import spacy
import srsly
from spacy.language import Language
from spacy.scorer import Scorer
from tqdm import tqdm, tqdm_notebook
from wasabi import Printer

from .constants import NONE
from .recognizer import EntityRecognizer
from .types import (
    Example,
    HardestExample,
    LabelDisparity,
    PredictionError,
    PredictionErrorExamplePair,
    Span,
)


def get_ents_by_label(
    data: List[Example], use_lower: bool = True
) -> DefaultDict[str, List[str]]:
    """Get a dictionary of unique text spans by label for your data
    
    Args:
        data (List[Example]): List of Examples
        use_lower (bool, optional): Use the lowercase form of the span text.
    
    Returns:
        DefaultDict[str, List[str]]: DefaultDict mapping label to sorted list of the unique
            spans annotated for that label.
    """
    annotations: DefaultDict[str, Set[str]] = defaultdict(set)
    sorted_annotations: DefaultDict[str, List[str]] = defaultdict(list)

    for e in data:
        for s in e.spans:
            span_text = s.text.lower() if use_lower else s.text
            annotations[s.label].add(span_text)

    for label in annotations.keys():
        sorted_annotations[label] = sorted(annotations[label])

    return sorted_annotations


def get_label_disparities(
    data: List[Example], label1: str, label2: str, use_lower: bool = True
) -> Set[str]:
    """Identify annotated spans that have different labels in different examples
    
    Args:
        data (List[Example]): Input List of Examples
        label1 (str): First label to compare
        label2 (str): Second label to compare
        use_lower (bool, optional): Use the lowercase form of the span text in ents_to_label.
    
    Returns:
        Set[str]: Set of all unique text spans that overlap between label1 and label2
    """
    annotations = get_ents_by_label(data, use_lower=use_lower)
    return set(annotations[label1]).intersection(set(annotations[label2]))


def top_label_disparities(
    data: List[Example], use_lower: bool = True, dedupe: bool = False
) -> List[LabelDisparity]:
    """Identify annotated spans that have different labels
    in different examples for all label pairs in data.
    
    Args:
        data (List[Example]): Input List of Examples
        use_lower (bool, optional): Use the lowercase form of the span text in ents_to_label.
        dedupe (bool, optional): Whether to deduplicate for table view vs confusion matrix.
            False by default for easy confusion matrix display.
    
    Returns:
        List[LabelDisparity]: List of LabelDisparity objects for each label pair combination
            sorted by the number of disparities between them.
    """
    annotations = get_ents_by_label(data, use_lower=use_lower)
    label_disparities = {}
    for label1 in annotations.keys():
        for label2 in annotations.keys():
            if label1 != label2:
                n_disparities = len(
                    set(annotations[label1]).intersection(set(annotations[label2]))
                )
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
    labels: List[str] = None,
    n: int = None,
    k: int = None,
    exclude_fp: bool = False,
    exclude_fn: bool = False,
    verbose: bool = False,
) -> List[PredictionError]:
    """Get a sorted list of examples your model is worst at predicting.
    
    Args:
        recognizer (EntityRecognizer): An instance of EntityRecognizer
        data (List[Example]): List of annotated Examples
        labels (List[str], optional): List of labels to get errors for. 
            Defaults to the labels property of `recognizer`.
        n (int, optional): If set, only use the top n examples from data.
        k (int, optional): If set, return the top k prediction errors, otherwise the whole list.
        exclude_fp (bool, optional): Flag to exclude False Positive errors.
        exclude_fn (bool, optional): Flag to exclude False Negative errors.
        verbose (bool, optional): Show verbose output.
    
    Returns:
        List[PredictionError]: List of Prediction Errors your model is making, sorted by the
            spans your model has the most trouble with.
    """
    labels_ = labels or recognizer.labels
    if n is not None:
        data = data[:n]

    n_examples = len(data)
    texts = (e.text for e in data)
    anns = (e.spans for e in data)

    errors = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # type: ignore
    error_examples: DefaultDict[str, List[PredictionErrorExamplePair]] = defaultdict(
        list
    )
    n_errors = 0

    for orig_example, pred_example, ann in zip(data, recognizer.predict(texts), anns):
        if k is not None and n_errors > k:
            break

        pred_error_example_pair = PredictionErrorExamplePair(
            original=orig_example, predicted=pred_example
        )

        cand = set([(s.start, s.end, s.label) for s in pred_example.spans])
        gold = set([(s.start, s.end, s.label) for s in ann])

        fp_diff = cand - gold
        fn_diff = gold - cand

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
                else:
                    start, end, false_label = fp
                    text = pred_example.text[start:end]
                    errors[NONE][text][false_label] += 1
                error_examples[text].append(pred_error_example_pair)
                n_errors += 1

        if fn_diff and not exclude_fn:
            for fn in fn_diff:
                start, end, label = fn
                text = pred_example.text[start:end]
                errors[label][text][NONE] += 1
                error_examples[text].append(pred_error_example_pair)
                n_errors += 1

    ranked_errors: List[PredictionError] = []

    for label, errors_per_label in errors.items():
        for error_text, error_labels in errors_per_label.items():
            for error_label, count in error_labels.items():
                ranked_errors.append(
                    PredictionError(
                        text=error_text,
                        true_label=label,
                        pred_label=error_label,
                        count=count,
                        examples=error_examples[error_text],
                    )
                )

    ranked_errors = sorted(ranked_errors, key=lambda error: error.count, reverse=True)
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
    pred_errors: List[PredictionError],
    return_pred_errors: bool = True,
    remove_pred_error_examples: bool = True,
) -> List[HardestExample]:
    """Get hardest examples from list of PredictionError types
    
    Args:
        pred_errors (List[PredictionError]): list of PredictionError
        return_pred_errors (bool, optional): Whether to return prediction errors. Defaults to True.
        remove_pred_error_examples (bool, optional): Whether to remove examples from returned PredictionError. Defaults to True.
    
    Raises:
        ValueError: Each PredictionError must have a List of Examples 
    
    Returns:
        List[HardestExample]: Sorted list of the hardest examples for a model to work on.
    """

    has_examples = any([pe.examples for pe in pred_errors])
    if not has_examples:
        raise ValueError(
            "Each PredictionError in Parameter pred_errors must have examples attached."
        )

    examples_text_map: Dict[str, Example] = {}
    example_pred_errors_map: DefaultDict[str, List[PredictionError]] = defaultdict(list)
    for pe in pred_errors:
        if pe.examples:
            for example in pe.examples:
                examples_text_map[example.original.text] = example.original
                example_pred_errors_map[example.original.text].append(pe)

    hardest_examples = []
    for example_text, example_pred_errors in example_pred_errors_map.items():
        example = examples_text_map[example_text]  # type: ignore

        record = HardestExample(example=example, count=len(example_pred_errors))
        if return_pred_errors:
            if remove_pred_error_examples:
                _example_pred_errors = copy.deepcopy(example_pred_errors)
                for _pe in _example_pred_errors:
                    _pe.examples = []

                record.prediction_errors = _example_pred_errors
            else:
                record.prediction_errors = example_pred_errors
        hardest_examples.append(record)

    sorted_hardest_examples = sorted(
        hardest_examples, key=lambda he: he.count, reverse=True
    )
    return sorted_hardest_examples
