from typing import List

from recon.operations.core import operation
from recon.types import Example, Span


@operation("recon.upcase_labels.v1")
def upcase_labels(example: Example) -> Example:
    """Convert all span labels to uppercase to normalize

    Args:
        example (Example): Input Example

    Returns:
        Example: Example with fixed labels
    """
    for s in example.spans:
        s.label = s.label.upper()
    return example


@operation("recon.filter_overlaps.v1")
def filter_overlaps(example: Example) -> Example:
    """Filter overlapping entity spans by picking the longest one.

    Args:
        example (Example): Input Example

    Returns:
        List[Example]: Example with fixed overlaps
    """
    annotations: List[Span] = sorted(example.spans, key=lambda s: s.start)
    filtered_annotations = remove_overlapping_entities(annotations)
    example.spans = filtered_annotations

    return example


def select_subset_of_overlapping_chain(chain: List[Span]) -> List[Span]:
    """
    Select the subset of entities in an overlapping chain to return by greedily choosing the
    longest entity in the chain until there are no entities remaining
    """
    sorted_chain = sorted(chain, key=lambda s: s.end - s.start, reverse=True)
    selections_from_chain: List[Span] = []
    chain_index = 0
    # dump the current chain by greedily keeping the longest entity that doesn't overlap
    while chain_index < len(sorted_chain):
        entity = sorted_chain[chain_index]
        match_found = False
        for already_selected_entity in selections_from_chain:
            max_start = max(entity.start, already_selected_entity.start)
            min_end = min(entity.end, already_selected_entity.end)
            if len(range(max_start, min_end)) > 0:
                match_found = True
                break

        if not match_found:
            selections_from_chain.append(entity)

        chain_index += 1

    return selections_from_chain


def remove_overlapping_entities(sorted_spans: List[Span]) -> List[Span]:
    """
    Removes overlapping entities from the entity set, by greedilytaking the longest
    entity from each overlapping chain. The input list of entities should be sorted
    and follow the spacy format.
    """
    spans_without_overlap: List[Span] = []
    current_overlapping_chain: List[Span] = []
    current_overlapping_chain_start = 0
    current_overlapping_chain_end = 0
    for i, current_entity in enumerate(sorted_spans):
        current_entity = sorted_spans[i]
        current_entity_start = current_entity.start
        current_entity_end = current_entity.end

        if len(current_overlapping_chain) == 0:
            current_overlapping_chain.append(current_entity)
            current_overlapping_chain_start = current_entity_start
            current_overlapping_chain_end = current_entity_end
        else:
            min_end = min(current_entity_end, current_overlapping_chain_end)
            max_start = max(current_entity_start, current_overlapping_chain_start)
            if min_end - max_start > 0:
                current_overlapping_chain.append(current_entity)
                current_overlapping_chain_start = min(
                    current_entity_start, current_overlapping_chain_start
                )
                current_overlapping_chain_end = max(
                    current_entity_end, current_overlapping_chain_end
                )
            else:
                selections_from_chain: List[Span] = select_subset_of_overlapping_chain(
                    current_overlapping_chain
                )

                current_overlapping_chain = []
                spans_without_overlap.extend(selections_from_chain)
                current_overlapping_chain.append(current_entity)
                current_overlapping_chain_start = current_entity_start
                current_overlapping_chain_end = current_entity_end

    spans_without_overlap.extend(select_subset_of_overlapping_chain(current_overlapping_chain))

    return sorted(spans_without_overlap, key=lambda x: x.start)
