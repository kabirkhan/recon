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
from .types import EntityCoverage, Example, LabelDisparity, PredictionError


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
    data: List[Example], use_lower: bool = True
) -> List[LabelDisparity]:
    """Identify annotated spans that have different labels
    in different examples for all label pairs in data.
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        Input List of Examples
    **use_lower**: (bool, optional), Defaults to True.
        Use the lowercase form of the span text in ents_to_label
    
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
                    input_hash = "||".join(sorted([label1, label2]))
                    label_disparities[input_hash] = {
                        "label1": label1,
                        "label2": label2,
                        "n_disparities": n_disparities,
                    }

    return sorted(
        label_disparities.values(), key=lambda row: row["n_disparities"], reverse=True
    )


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
        List of Prediction Errors your model is making sorted by the
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

    for orig_example, pred, ann in zip(data, ner.predict(texts), anns):

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

        if fn_diff and not exclude_fn:
            for fn in fn_diff:
                start, end, label = fn
                text = pred.text[start:end]
                errors[label][text][NONE] += 1
                error_examples[text].append(orig_example)

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
    if k:
        ranked_errors = ranked_errors[:k]
    error_texts = set()
    for re in ranked_errors:
        for e in re.examples:
            error_texts.add(e.text)
            
    error_rate = round(len(error_texts) / len(data), 2)
    if verbose:
        error_summary = {
            "N Examples": len(data),
            "N Errors": len(ranked_errors),
            "N Error Examples": len(error_texts),
            "Error Rate": error_rate
        }
        msg = Printer()
        msg.divider("Error Analysis")
        msg.table(error_summary)
    
    return ranked_errors


def entity_coverage(data: List[Example],
                    sep: str = "||",
                    use_lower: bool = True,
                    return_examples: bool = False) -> List[EntityCoverage]:
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
    coverage_map = defaultdict(int)
    examples_map = defaultdict(list)

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
            record['examples'] = examples_map[key]
        coverage.append(record)

    sorted_coverage = sorted(coverage, key=lambda x: x['count'], reverse=True)
    return sorted_coverage
