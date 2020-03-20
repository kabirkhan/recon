"""Make corrections to your data."""

import copy
from collections import defaultdict
from typing import DefaultDict, Dict, List

from .types import Example


def rename_labels(data: List[Example], label_map: Dict[str, str]) -> List[Example]:
    """Rename labels in a copy of List[Example] data
    
    Args:
        data (List[Example]): List of Examples
        label_map (Dict[str, str]): One-to-one mapping of label names
    
    Returns:
        List[Example]: Copy List of Examples with renamed labels
    """
    data_copy = copy.deepcopy(data)
    for example in data_copy:
        for span in example.spans:
            span.label = label_map.get(span.label, span.label)
    return data_copy


def fix_annotations(
    data: List[Example], corrections: Dict[str, str], use_lower: bool = True
) -> List[Example]:
    """Fix annotations in a copy of List[Example] data.
    
    This function will NOT add annotations to your data.
    It will only remove erroneous annotations and fix the
    labels for specific spans.
    
    Args:
        data (List[Example]): List of Examples
        corrections (Dict[str, str]): Dictionary of corrections mapping entity text to a new label.
        If the value is set to None, the annotation will be removed
        use_lower (bool, optional): Use the lowercase form of the span text 
            for matching corrections. Defaults to True.
    
    Returns:
        List[Example]: Fixed Examples
    """
    data_copy: List[Example] = copy.deepcopy(data)
    if use_lower:
        corrections = {t.lower(): l for t, l in corrections.items()}

    prints: DefaultDict[str, List[str]] = defaultdict(list)

    for example in data:
        ents_to_remove = []
        for i, s in enumerate(example.spans):
            t = s.text.lower()

            if t in corrections:
                if corrections[t] is print:
                    prints[t] += [("=" * 100), example.text, s.label]
                elif corrections[t] is None:
                    ents_to_remove.append(i)
                else:
                    s.label = corrections[t]

        i = len(ents_to_remove) - 1
        while i >= 0:
            idx = ents_to_remove[i]
            del example.spans[idx]
            i -= 1

    for k in sorted(prints):
        print(f"**{k}**")
        for line in prints[k]:
            print(line)

    return data
