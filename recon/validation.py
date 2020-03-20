from typing import Any, Dict, List, Tuple

from spacy.language import Language

from .types import Example


def json_to_examples(data: List[Dict[str, Any]]) -> List[Example]:
    """Convert List of Dicts to List of typed Examples
    
    Args:
        data (List[Dict[str, Any]]): Input List of Dicts to convert
    
    Returns:
        List[Example]: List of typed Examples
    """
    return [Example(**example) for example in data]


def fix_annotations_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix annotations format for a consistent dataset
    
    Args:
        data (List[Dict[str, Any]]): List of JSON Examples
    
    Returns:
        List[Dict[str, Any]]: List of JSON Examples with corrected formatting
    """
    for e in data:
        if "meta" not in e:
            e["meta"] = {}
        if isinstance(e["meta"], list) or isinstance(e["meta"], str):
            e["meta"] = {"source": e["meta"]}

        for s in e["spans"]:
            if "text" not in s:
                s["text"] = e["text"][s["start"] : s["end"]]
            s["label"] = s["label"].upper()
    return data


def add_tokens(nlp: Language, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add tokens to each to JSON Example
    
    Args:
        nlp (Language): spaCy Language instance for tokenization
        data (List[Dict[str, Any]]): List of JSON Examples
    
    Returns:
        List[Dict[str, Any]]: List of JSON Examples with tokens
    """
    texts = (e["text"] for e in data)

    with nlp.disable_pipes(*nlp.pipe_names):
        for e, doc in zip(data, nlp.pipe(texts)):
            e["tokens"] = [
                {"text": t.text, "start": t.idx, "end": t.idx + len(t), "id": t.i}
                for t in doc
            ]

    return data


def filter_overlaps(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter overlapping entity spans by picking the longest one.
    
    Args:
        data (List[Dict[str, Any]]): List of Examples
    
    Returns:
        List[Dict[str, Any]]: List of Examples with fixed overlaps
    """
    out_data = []
    for e in data:
        annotations = sorted(e["spans"], key=lambda x: x["start"])
        filtered_annotations = remove_overlapping_entities(annotations)

        new_e = e.copy()
        new_e["spans"] = filtered_annotations
        out_data.append(new_e)

    return out_data


def select_subset_of_overlapping_chain(
    chain: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Select the subset of entities in an overlapping chain to return by greedily choosing the
    longest entity in the chain until there are no entities remaining
    """
    sorted_chain = sorted(chain, key=lambda x: x["end"] - x["start"], reverse=True)
    selections_from_chain: List[Dict[str, Any]] = []
    chain_index = 0
    # dump the current chain by greedily keeping the longest entity that doesn't overlap
    while chain_index < len(sorted_chain):
        entity = sorted_chain[chain_index]
        match_found = False
        for already_selected_entity in selections_from_chain:
            max_start = max(entity["start"], already_selected_entity["start"])
            min_end = min(entity["end"], already_selected_entity["end"])
            if len(range(max_start, min_end)) > 0:
                match_found = True
                break

        if not match_found:
            selections_from_chain.append(entity)

        chain_index += 1

    return selections_from_chain


def remove_overlapping_entities(
    sorted_spacy_format_entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Removes overlapping entities from the entity set, by greedilytaking the longest
    entity from each overlapping chain. The input list of entities should be sorted
    and follow the spacy format.
    """
    spacy_format_entities_without_overlap = []
    current_overlapping_chain: List[Dict[str, Any]] = []
    current_overlapping_chain_start = 0
    current_overlapping_chain_end = 0
    for i, current_entity in enumerate(sorted_spacy_format_entities):
        current_entity = sorted_spacy_format_entities[i]
        current_entity_start = current_entity["start"]
        current_entity_end = current_entity["end"]

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
                selections_from_chain: List[
                    Dict[str, Any]
                ] = select_subset_of_overlapping_chain(current_overlapping_chain)

                current_overlapping_chain = []
                spacy_format_entities_without_overlap.extend(selections_from_chain)
                current_overlapping_chain.append(current_entity)
                current_overlapping_chain_start = current_entity_start
                current_overlapping_chain_end = current_entity_end

    spacy_format_entities_without_overlap.extend(
        select_subset_of_overlapping_chain(current_overlapping_chain)
    )

    return sorted(spacy_format_entities_without_overlap, key=lambda x: x["start"])
