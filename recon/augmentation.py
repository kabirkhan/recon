from typing import Any, Callable, Dict, List, Optional

import numpy as np
from recon.operations import operation
from recon.types import Example, Span


def mask_1d(length: int, prob: float = 0.5) -> np.ndarray:
    if prob < 0 or prob > 1:
        raise ValueError(f"Prob of {prob} is not allowed. Allowed values between 0 and 1.")

    mask = np.zeros(length, dtype=int)
    mask[: np.ceil(length * prob).astype(int)] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool)
    return mask


def substitute_spans(example: Example, span_subs: Dict[int, str]) -> Example:
    """Substitute spans in an example. Replaces span text and alters the example text
    and span offsets to create a valid example.

    Args:
        example (Example): Input example
        span_subs (Dict[int, str]): Mapping of span hash to a str replacement text

    Returns:
        Example: Output example with substituted spans
    """
    span_sub_start_counter = 0

    new_example_text = example.text

    for span in example.spans:
        prev_end = span.end
        new_text = span.text

        span_hash = hash(span)
        if span_hash in span_subs:
            new_text = span_subs[span_hash]
            new_start = span.start + span_sub_start_counter
            new_end = new_start + len(new_text)

            new_example_text = (
                new_example_text[: span.start + span_sub_start_counter]
                + new_text
                + new_example_text[span.end + span_sub_start_counter :]
            )

            span_sub_start_counter = new_end - prev_end
            span.text = new_text
            span.start = new_start
            span.end = new_end
        else:
            span.start += span_sub_start_counter
            span.end = span.start + len(new_text)
            span_sub_start_counter = span.end - prev_end

        span.text = new_text
    example.text = new_example_text

    return example


def augment_example(
    example: Example,
    span_f: Callable[[Span, Any], Optional[str]],
    span_label: str = None,
    n_augs: int = 1,
    sub_prob: float = 0.5,
    **kwargs: Any,
) -> List[Example]:

    prev_example = example.copy(deep=True)

    augmented_examples = [prev_example.copy(deep=True)]
    augmented_example_hashes = {hash(prev_example)}

    for i in range(n_augs):
        example = prev_example.copy(deep=True)
        spans = example.spans
        if span_label:
            spans = [s for s in example.spans if s.label == span_label]
        mask = mask_1d(len(spans), prob=sub_prob)
        spans_to_sub = list(np.asarray(spans)[mask])

        span_subs = {}
        for span in spans_to_sub:
            res = span_f(span, **kwargs)  #  type: ignore
            if res:
                span_subs[hash(span)] = res

        if not any(span_subs.values()) or len(augmented_examples) > n_augs:
            break

        example = substitute_spans(example, span_subs)
        if hash(example) not in augmented_example_hashes:
            augmented_examples.append(example)
            augmented_example_hashes.add(hash(example))

    return augmented_examples


@operation("recon.v1.augment.ent_label_sub", handles_tokens=False)
def ent_label_sub(
    example: Example, label: str, subs: List[str], n_augs: int = 1, sub_prob: float = 0.5
) -> List[Example]:
    """Augmentation to substitute entities based on label.
    Applies a mask to the entities that have a provided label based on substitution_prob
    and selects a random choice from the list of provided substitutions to replace each
    span with

    Args:
        example (Example): Input example
        label (str): Span label to replace
            e.g. PERSON or LOCATION
        subs (List[str]): List of substitutions
            e.g. list of names if label is PERSON
        n_augs (int, optional): Maximum number of augmentated examples to create per example.
        sub_prob (float, optional): Probability from 0-1 inclusive of how many of the spans to replace

    Returns:
        List[Example]: List of augmented examples including the original.
    """

    def augmentation(span: Span, subs: List[str]) -> Optional[str]:
        subs = [s for s in subs if s != span.text]
        sub = None
        if len(subs) > 0:
            sub = np.random.choice(subs)
        return sub

    return augment_example(example, augmentation, n_augs=n_augs, sub_prob=sub_prob, subs=subs)


@operation("recon.v1.augment.kb_expansion", factory=True)
def kb_expansion(
    example: Example,
    preprocessed_outputs: Dict[str, Any] = {},
    n_augs: int = 1,
    sub_prob: float = 0.5,
) -> List[Example]:

    spans_to_aliases_map = preprocessed_outputs["recon.v1.span_aliases"]

    def augmentation(span: Span, spans_to_aliases_map: Dict[int, List[str]]) -> Optional[str]:
        sub = None
        if hash(span) in spans_to_aliases_map:
            aliases = spans_to_aliases_map[hash(span)]

            if len(aliases) > 0:
                rand_alias = np.random.choice(aliases)
                index = aliases.index(rand_alias)
                del spans_to_aliases_map[hash(span)][index]
                sub = rand_alias

        return sub

    return augment_example(
        example,
        augmentation,
        n_augs=n_augs,
        sub_prob=sub_prob,
        spans_to_aliases_map=spans_to_aliases_map,
    )
