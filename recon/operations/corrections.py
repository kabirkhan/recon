"""Make corrections to your data."""

from typing import Any, Dict, List, cast

from spacy.tokens import Span as SpacySpan
from wasabi import msg

from recon.operations.core import operation
from recon.types import Correction, Example, Span, Token


@operation("recon.rename_labels.v1")
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


@operation("recon.fix_annotations.v1")
def fix_annotations(
    example: Example,
    corrections: List[Correction],
    case_sensitive: bool = False,
    dryrun: bool = False,
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
        dryrun (bool, optional): Treat corrections as a dryrun and just print all changes to be made

    Returns:
        Example: Example with fixed annotations
    """

    if not case_sensitive:
        for c in corrections:
            c.annotation = c.annotation.lower()

    corrections_map: Dict[str, Correction] = {c.annotation: c for c in corrections}
    prints: List[str] = []

    ents_to_remove: List[int] = []
    for i, s in enumerate(example.spans):
        t = s.text if case_sensitive else s.text.lower()

        if t in corrections_map:
            c = corrections_map[t]
            if c.to_label is None and (s.label in c.from_labels or "ANY" in c.from_labels):
                if dryrun:
                    prints.append(f"Deleting span: {s.text}")
                else:
                    ents_to_remove.append(i)
            elif s.label in c.from_labels or "ANY" in c.from_labels:
                if dryrun:
                    prints.append(
                        f"Correction span: {s.text} from labels: {c.from_labels} to label: {c.to_label}"
                    )
                else:
                    s.label = cast(str, c.to_label)

    i = len(ents_to_remove) - 1
    while i >= 0:
        idx = ents_to_remove[i]
        del example.spans[idx]
        i -= 1

    if dryrun:
        msg.divider("Example Text")
        msg.text(example.text)
        for line in prints:
            msg.text(line)

    return example


def corrections_from_dict(corrections_dict: Dict[str, Any]) -> List[Correction]:
    """Create a list of Correction objects from a simpler config for
    corrections using a Dict representation mapping keys to either the label to
    convert to or a tuple of (from_label, to_label) pairings or (List[from_labels], to_label)
    pairings if you want to convert as subset of labels at a time

    Args:
        corrections_dict (Dict[str, Any]): Corrections formatted dict
            e.g. {
                "united states": "GPE",
                "London": (["LOC"], "GPE")
            }

    Raises:
        ValueError: If the format of the dict

    Returns:
        [type]: [description]
    """
    corrections: List[Correction] = []
    for key, val in corrections_dict.items():
        if isinstance(val, str) or val is None:
            from_labels = ["ANY"]
            to_label = val
        elif isinstance(val, tuple):
            if isinstance(val[0], str):
                from_labels = [val[0]]
            else:
                from_labels = val[0]
            to_label = val[1]
        else:
            raise ValueError(
                "Cannot parse corrections dict. Value must be either a str of the label "
                + "to change the annotation to (TO_LABEL) or a tuple of (FROM_LABEL, TO_LABEL)"
            )
        corrections.append(Correction(annotation=key, from_labels=from_labels, to_label=to_label))
    return corrections


@operation("recon.strip_annotations.v1", pre=["recon.spacy.v1"], handles_tokens=False)
def strip_annotations(
    example: Example,
    *,
    strip_chars: List[str] = [".", "!", "?", "-", ":", " "],
    preprocessed_outputs: Dict[str, Any] = {},
) -> Example:
    """Strip punctuation and spaces from start and end of annotations.
    These characters are almost always a mistake and will confuse a model

    Args:
        example (Example): Input Example
        strip_chars (List[str], optional): Characters to strip.

    Returns:
        Example: Example with stripped spans
    """

    preprocessed_outputs["recon.spacy.v1"]

    def fix_tokens(example: Example, span: Span) -> bool:
        fix_tokens = example.tokens and any(example.tokens)
        return bool(
            fix_tokens and span.token_start and span.token_end and span.token_start < span.token_end
        )

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


@operation("recon.split_sentences.v1", pre=["recon.spacy.v1"])
def split_sentences(example: Example, preprocessed_outputs: Dict[str, Any] = {}) -> List[Example]:
    """Split a single example into multiple examples by splitting the text into
    multiple sentences and resetting entity and token offsets based on offsets
    relative to sentence boundaries

    Args:
        example (Example): Input Example
        preprocessed_outputs (Dict[str, Any], optional): Outputs of preprocessors.

    Returns:
        List[Example]: List of split examples.
            Could be list of 1 if the example is just one sentence.
    """
    doc = preprocessed_outputs["recon.spacy.v1"]

    new_examples = []
    ents = []
    for ent in example.spans:
        span = doc.char_span(ent.start, ent.end, label=ent.label)
        if not span:
            token = None
            text = doc.text[ent.start : ent.end]
            for t in doc:
                if t.text == text:
                    token = t
            if token:
                span = SpacySpan(doc, token.i, token.i + 1, label=ent.label)
        ents.append(span)

    doc.ents = ents

    for sent in doc.sents:
        sent_doc = sent.as_doc()
        new_example = Example(
            text=sent.text,
            spans=[
                Span(
                    text=e.text,
                    start=e.start_char,
                    end=e.end_char,
                    token_start=e.start,
                    token_end=e.end,
                    label=e.label_,
                )
                for e in sent_doc.ents
            ],
            tokens=[
                Token(text=t.text, start=t.idx, end=t.idx + len(t.text), id=i)
                for i, t in enumerate(sent_doc)
            ],
        )
        new_examples.append(new_example)
    return new_examples
