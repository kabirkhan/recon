from typing import Any, Callable, Dict, List, Optional

import numpy as np

from recon.operations import operation
from recon.types import Example, Span


def mask_1d(length: int, prob: float = 0.5) -> np.ndarray:
    if prob < 0 or prob > 1:
        raise ValueError(
            f"Prob of {prob} is not allowed. Allowed values between 0 and 1."
        )

    mask = np.zeros(length, dtype=int)
    mask[: np.ceil(length * prob).astype(int)] = 1
    np.random.shuffle(mask)
    mask = mask.astype(bool)
    return mask


def substitute_spans(example: Example, span_subs: Dict[Span, str]) -> Example:
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
    new_example_spans = []

    prev_example_spans = {hash(span) for span in example.spans}
    spans = sorted(set(list(span_subs.keys()) + example.spans), key=lambda s: s.start)

    for span in spans:
        should_add_span = hash(span) in prev_example_spans

        prev_end = span.end
        new_text = span.text

        if span in span_subs:
            new_text = span_subs[span]
            new_start = span.start + span_sub_start_counter
            new_end = new_start + len(new_text)

            new_example_text = (
                new_example_text[: span.start + span_sub_start_counter]
                + new_text
                + new_example_text[span.end + span_sub_start_counter :]
            )

            span.text = new_text
            span.start = new_start
            span.end = new_end

            span_sub_start_counter += new_end - prev_end
        else:
            span.start += span_sub_start_counter
            span.end = span.start + len(new_text)
            span_sub_start_counter = span.end - prev_end

        span.text = new_text

        if should_add_span:
            new_example_spans.append(span)

    example.text = new_example_text
    example.spans = new_example_spans

    return example


def augment_example(
    example: Example,
    span_f: Callable[[Span, Any], Optional[str]],
    spans: Optional[List[Span]] = None,
    span_label: Optional[str] = None,
    n_augs: int = 1,
    sub_prob: float = 0.5,
    **kwargs: Any,
) -> List[Example]:
    if spans is None:
        spans = example.model_copy(deep=True).spans

    augmented_examples = {example}

    for _ in range(n_augs):
        example = example.model_copy(deep=True)
        if span_label:
            spans = [s for s in spans if s.label == span_label]
        mask = mask_1d(len(spans), prob=sub_prob)
        spans_to_sub = list(np.asarray(spans)[mask])

        span_subs = {}
        for span in spans_to_sub:
            res = span_f(span, **kwargs)  # type: ignore
            if res:
                span_subs[span] = res

        if not any(span_subs.values()) or len(augmented_examples) > n_augs:
            break

        example = substitute_spans(example, span_subs)
        if example not in augmented_examples:
            augmented_examples.add(example)

    return list(augmented_examples)


@operation("recon.augment.ent_label_sub.v1", handles_tokens=False)
def ent_label_sub(
    example: Example,
    label: str,
    subs: List[str],
    n_augs: int = 1,
    sub_prob: float = 0.5,
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
        n_augs (int, optional): Maximum number of augmentated examples to
            create per example.
        sub_prob (float, optional): Probability from 0-1
            inclusive of how many of the spans to replace

    Returns:
        List[Example]: List of augmented examples including the original.
    """

    def augmentation(span: Span, subs: List[str]) -> Optional[str]:
        subs = [s for s in subs if s != span.text]
        sub = None
        if len(subs) > 0:
            sub = np.random.choice(subs)
        return sub

    return augment_example(
        example,
        augmentation,
        span_label=label,
        n_augs=n_augs,
        sub_prob=sub_prob,
        subs=subs,
    )


@operation("recon.augment.kb_expansion.v1", factory=True)
def kb_expansion(
    example: Example,
    preprocessed_outputs: Dict[str, Any] = {},
    n_augs: int = 1,
    sub_prob: float = 0.5,
) -> List[Example]:
    spans_to_aliases_map = preprocessed_outputs["recon.span_aliases.v1"]

    def augmentation(
        span: Span, spans_to_aliases_map: Dict[int, List[str]]
    ) -> Optional[str]:
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
