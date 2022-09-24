import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from recon.types import Example


def hash_example_meta(
    example: Example, fields: List[str] = [], ignore_field_absence: bool = False
) -> Tuple:
    """Create a hash out of the metadata of an example

    Args:
        example (Example): Input Example
        fields (List[str]): Meta fields to use in hash. Defaults to all availabile fields.
        ignore_field_absence (bool, optional): Determines behavior of when a field does not exist
            in an example's meta property

    Raises:
        ValueError: If a field passed to fields does not exist in an example's meta

    Returns:
        Tuple: Tuple of hashable attributes from the meta of an example
    """
    if not fields:
        fields = list(example.meta.keys())

    tpl = []
    for field in fields:
        tpl.append(field)
        if field not in example.meta:
            if ignore_field_absence:
                continue
            raise ValueError(f"Field {field} not present in 'meta' for example {example}")
        meta_val = example.meta[field]
        if isinstance(meta_val, list):
            tpl += meta_val
        else:
            tpl.append(meta_val)

    return tuple(tpl)


def sample_examples(
    examples: List[Example],
    meta_filters: Dict[str, List[str]] = {},
    fields: List[str] = [],
    ignore_field_absence: bool = False,
    top_k_per_hash: int = 10,
    top_k: int = -1,
    shuffle: bool = True,
) -> List[Example]:
    """Sample examples based on meta attributes

    Args:
        examples (List[Example]): Examples to sample from
        meta_filters (Dict[str, List[str]], optional): Values to filter out of sampled set for each meta field.
        fields (List[str], optional): Meta fields to use in hash. Defaults to all availabile fields.
        ignore_field_absence (bool, optional): Determines behavior of when a field does not exist
            in an example's meta property.
        top_k_per_hash (int, optional): Number of examples to include per meta hash.
        top_k (int, optional): Total number of examples to sample from.

    Returns:
        List[Example]: Sampled examples
    """

    examples_counter: Dict[Any, int] = defaultdict(int)
    out_examples = []

    if shuffle:
        random.shuffle(examples)

    for example in examples:

        if top_k > 0 and sum(examples_counter.values()) >= top_k:
            break

        for meta_field, meta_vals in meta_filters.items():
            if example.meta[meta_field] not in meta_vals:
                break

        meta_hash = hash_example_meta(
            example, fields=fields, ignore_field_absence=ignore_field_absence
        )
        if examples_counter[meta_hash] <= top_k_per_hash:
            out_examples.append(example)
            examples_counter[meta_hash] += 1

    return out_examples
