"""Make corrections to your data."""

import copy
from collections import defaultdict
from typing import DefaultDict, Dict, List

from .operations import operation
from .types import Example, TransformationCallbacks


@operation("recon.v1.rename_labels")
def rename_labels(example: Example, label_map: Dict[str, str]) -> Example:
    """Rename labels in a copy of List[Example] data
    
    Args:
        example (Example): Input Example
        label_map (Dict[str, str]): One-to-one mapping of label names
    
    Returns:
        Example: Copy of Example with renamed labels
    """
    for span in example.spans:
        span.label = label_map.get(span.label, span.label)
    return example


@operation("recon.v1.fix_annotations")
def fix_annotations(
    example: Example, corrections: Dict[str, str], case_sensitive: bool = False
) -> Example:
    """Fix annotations in a copy of List[Example] data.
    
    This function will NOT add annotations to your data.
    It will only remove erroneous annotations and fix the
    labels for specific spans.
    
    Args:
        example (Example): Input Example
        corrections (Dict[str, str]): Dictionary of corrections mapping entity text to a new label.
            If the value is set to None, the annotation will be removed
        case_sensitive (bool, optional): Consider case of text for each correction
    
    Returns:
        Example: Example with fixed annotations
    """
    if case_sensitive:
        corrections = {t: l for t, l in corrections.items()}
    else:
        corrections = {t.lower(): l for t, l in corrections.items()}

    prints: DefaultDict[str, List[str]] = defaultdict(list)

    ents_to_remove = []
    for i, s in enumerate(example.spans):
        t = s.text if case_sensitive else s.text.lower()

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

    return example


@operation("recon.v1.strip_annotations")
def strip_annotations(
    example: Example, strip_chars: List[str] = [".", "!", "?", "-", ":", " "]
) -> Example:
    """Strip punctuation and spaces from start and end of annotations.
    These characters are almost always a mistake and will confuse a model
    
    Args:
        example (Example): Input Example
        strip_chars (List[str], optional): Characters to strip.
    
    Returns:
        Example: Example with stripped spans
    """

    for s in example.spans:
        for ch in strip_chars:
            if s.text.startswith(ch):
                ch = s.text[0]

                while ch in strip_chars:
                    s.text = s.text[1:]
                    s.start += 1
                    ch = s.text[0]
            elif s.text.endswith(ch):
                ch = s.text[-1]
                while ch in strip_chars:
                    s.text = s.text[:-1]
                    ch = s.text[-1]
                    s.end -= 1
    return example
