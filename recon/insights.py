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
    EntityCoverage,
    Example,
    HardestExample,
    LabelDisparity,
    PredictionError,
)


def ents_by_label(
    data: List[Example], use_lower: bool = True
) -> DefaultDict[str, List[str]]:
    """Get a dictionary of unique text spans by label for your data
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        List of Examples
    **use_lower**: (bool, optional), Defaults to True.
        Use the lowercase form of the span text
    
    ### Returns
    -----------
    (DefaultDict[str, List[str]]):
        DefaultDict mapping label to sorted list of the unique
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
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        Input List of Examples
    **label1**: (str), required.
        First label to compare
    **label2**: (str), required.
        Second label to compare
    **use_lower**: (bool, optional), Defaults to True.
        Use the lowercase form of the span text in ents_to_label
    
    ### Returns
    -----------
    (Set[str]): 
        Set of all unique text spans that overlap between label1 and label2
    """
    annotations = ents_by_label(data, use_lower=use_lower)
    return set(annotations[label1]).intersection(set(annotations[label2]))


def top_label_disparities(
    data: List[Example], use_lower: bool = True, dedupe: bool = False
) -> List[LabelDisparity]:
    """Identify annotated spans that have different labels
    in different examples for all label pairs in data.
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        Input List of Examples
    **use_lower**: (bool, optional), Defaults to True.
        Use the lowercase form of the span text in ents_to_label
    **dedupe**: (bool, optional), Defaults to False.
        Whether to deduplicate for table view vs confusion matrix.
        False by default for easy confusion matrix display
    
    ### Returns
    -----------
    (List[LabelDisparity]): 
        List of LabelDisparity objects for each label pair combination
        sorted by the number of disparities between them.
    """
    annotations = ents_by_label(data, use_lower=use_lower)
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
    ner: EntityRecognizer,
    data: List[Example],
    labels: List[str] = None,
    n: int = None,
    k: int = None,
    exclude_fp: bool = False,
    exclude_fn: bool = False,
    verbose: bool = False,
) -> List[PredictionError]:
    """Get a sorted list of examples your model is worst at predicting.
    
    ### Parameters
    --------------
    **ner**: (EntityRecognizer), required.
        An instance of EntityRecognizer
    **data**: (List[Example]), required.
        List of annotated Examples
    **labels**: (List[str], optional), Defaults to None.
        List of labels to get errors for. Defaults to the labels property of `ner`.
    **n**: (int, optional), Defaults to None.
        If set, only use the top n examples from data
    **k**: (int, optional), Defaults to None.
        If set, return the top k prediction errors, otherwise the whole list.
    **exclude_fp**: (bool, optional), Defaults to False.
        Flag to exclude False Positive errors.
    **exclude_fn**: (bool, optional), Defaults to False.
        Flag to exclude False Negative errors.
    **verbose**: (bool, optional), Defaults to False.
        Show progress_bar or not
    
    ### Returns
    -----------
    (List[PredictionError]): 
        List of Prediction Errors your model is making, sorted by the
        spans your model has the most trouble with.
    """

    labels_ = labels or ner.labels
    if n is not None:
        data = data[:n]

    n_examples = len(data)
    texts = (e.text for e in data)
    anns = (e.spans for e in data)

    errors = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # type: ignore
    error_examples: DefaultDict[str, List[Example]] = defaultdict(list)
    n_errors = 0

    for orig_example, pred, ann in zip(data, ner.predict(texts), anns):
        if k is not None and n_errors > k:
            break

        cand = set([(s.start, s.end, s.label) for s in pred.spans])
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
                    text = pred.text[start:end]
                    false_label = fp[2]
                    errors[label][text][false_label] += 1
                else:
                    start, end, false_label = fp
                    text = pred.text[start:end]
                    errors[NONE][text][false_label] += 1
                error_examples[text].append(orig_example)
                n_errors += 1

        if fn_diff and not exclude_fn:
            for fn in fn_diff:
                start, end, label = fn
                text = pred.text[start:end]
                errors[label][text][NONE] += 1
                error_examples[text].append(orig_example)
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
                error_texts.add(e.text)

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


def entity_coverage(
    data: List[Example],
    sep: str = "||",
    use_lower: bool = True,
    return_examples: bool = False,
) -> List[EntityCoverage]:
    """Identify how well you dataset covers an entity type. Get insights
    on the how many times certain text/label span combinations exist across your
    data so that you can focus your annotation efforts better rather than
    annotating examples your Model already understands well.
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        List of Examples
    **sep**: (str, optional), Defaults to "||".
        Separator used in coverage map, only change if || exists in your text
        or label
    **use_lower**: (bool, optional), Defaults to True.
        Use the lowercase form of the span text in ents_to_label
    **return_examples**: (bool, optional), Defaults to False.
        If True, return Examples that contain the entity label annotation.
    
    ### Returns
    -----------
    (List[EntityCoverage]): 
        Sorted List of EntityCoverage objects containing the text, label, count, and
        an optional list of examples where that text/label annotation exists.
        
        e.g.
        [
            {
                "text": "design",
                "label": "SKILL",
                "count": 243
            }
        ]
    """
    coverage_map: DefaultDict[str, int] = defaultdict(int)
    examples_map: DefaultDict[str, List[Example]] = defaultdict(list)

    for example in data:
        for span in example.spans:
            text = span.text
            if use_lower:
                text = text.lower()
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


def get_hardest_examples(
    pred_errors: List[PredictionError],
    return_pred_errors: bool = True,
    remove_pred_error_examples: bool = True,
) -> List[HardestExample]:
    """Get the hardest Examples given a list of PredictionErrors.
    Useful to run before streaming Examples back through Prodigy
    to prioritize these examples and ensure they're annotated correctly.
    
    ### Parameters
    --------------
    **pred_errors**: (List[PredictionError]), required.
        List of PredictionErrors
    **return_pred_errors**: (bool, optional), Defaults to True.
        Whether to return the PredictionErrors associated with each Example
    **remove_pred_error_examples**: (bool, optional), Defaults to True.
        If return_pred_errors is True, whether to remove the List of Examples
        for each PredictionError or not. Since you already have the Example
        this essentially just cleans up the output.
    
    ### Raises
    -----------
    ValueError: 
        If there are no examples present in the pred_errors Parameter
    
    ### Returns
    -----------
    (List[HardestExample]): 
        List of the HardestExample type that maps the Example to the number of 
        PredcitionErrors it contains as well as the optional list of PredcitionErrors
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
                examples_text_map[example.text] = example
                example_pred_errors_map[example.text].append(pe)

    hardest_examples = []
    for example_text, example_pred_errors in example_pred_errors_map.items():
        example = examples_text_map[example_text]

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
