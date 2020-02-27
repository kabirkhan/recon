from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Generator, List, Set, Union

import spacy
import srsly
from spacy.language import Language
from spacy.scorer import Scorer
from tqdm import tqdm, tqdm_notebook

from .constants import NONE
from .recognizer import EntityRecognizer
from .types import Example, PredictionError


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
) -> List[Dict[str, object]]:
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
    (): 
        [description]
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
        [description]
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
                for gold_ent in gold:
                    if fp[0] == gold_ent[0] and fp[1] == gold_ent[1]:
                        start, end, label = gold_ent
                        false_label = fp[2]
                        text = pred.text[start:end]
                        errors[label][text][false_label] += 1
                        error_examples[text].append(orig_example)

        if fn_diff and not exclude_fn:
            for fn in fn_diff:
                has_gold_ent = False
                for gold_ent in gold:
                    if fp[0] == gold_ent[0] and fp[1] == gold_ent[1]:
                        has_gold_ent = True

                if not has_gold_ent:
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
    return ranked_errors
