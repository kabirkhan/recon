"""Make corrections to your data."""

import copy
from collections import defaultdict
from typing import DefaultDict, Dict, List

from .operations import operation
from .types import Example, TransformationCallbacks


@operation("rename_labels")
def rename_labels(
    example: Example, label_map: Dict[str, str]
) -> List[Example]:
    """Rename labels in a copy of List[Example] data
    
    Args:
        data (List[Example]): List of Examples
        label_map (Dict[str, str]): One-to-one mapping of label names
    
    Returns:
        List[Example]: Copy List of Examples with renamed labels
    """
    for span in example.spans:
        span.label = label_map.get(span.label, span.label)
    return example


@operation("fix_annotations")
def fix_annotations(
    data: List[Example],
    corrections: Dict[str, str],
    *,
    case_sensitive: bool = False,
    callbacks: TransformationCallbacks = None,
) -> List[Example]:
    """Fix annotations in a copy of List[Example] data.
    
    This function will NOT add annotations to your data.
    It will only remove erroneous annotations and fix the
    labels for specific spans.
    
    Args:
        data (List[Example]): List of Examples
        corrections (Dict[str, str]): Dictionary of corrections mapping entity text to a new label.
        If the value is set to None, the annotation will be removed
        case_sensitive (bool, optional): Consider case of text for each correction
    
    Returns:
        List[Example]: Fixed Examples
    """
    examples: List[Example] = copy.deepcopy(data)
    if case_sensitive:
        corrections = {t: l for t, l in corrections.items()}
    else:
        corrections = {t.lower(): l for t, l in corrections.items()}

    prints: DefaultDict[str, List[str]] = defaultdict(list)

    for example in data:
        orig_example = hash(example)
        ents_to_remove = []
        for i, s in enumerate(example.spans):
            t = s.text if case_sensitive else s.text.lower()

            if t in corrections:
                print(t, "in corrections")
                if corrections[t] is print:
                    prints[t] += [("=" * 100), example.text, s.label]
                elif corrections[t] is None:
                    ents_to_remove.append(i)
                else:
                    print("before", s.label)
                    s.label = corrections[t]
                    print("after", s.label)

        i = len(ents_to_remove) - 1
        while i >= 0:
            idx = ents_to_remove[i]
            del example.spans[idx]
            i -= 1

        if hash(example) != orig_example:
            callbacks.change_example(orig_example, example)

    for k in sorted(prints):
        print(f"**{k}**")
        for line in prints[k]:
            print(line)

    return data
