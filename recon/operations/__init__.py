import catalogue

from .core import Operation, operation
from .corrections import (
    corrections_from_dict,
    fix_annotations,
    rename_labels,
    split_sentences,
    strip_annotations,
)
from .tokenization import add_tokens, fix_tokenization_and_spacing
from .validation import (
    filter_overlaps,
    remove_overlapping_entities,
    select_subset_of_overlapping_chain,
    upcase_labels,
)


class registry:
    operations = catalogue.create("recon", "operations", entry_points=True)
    operation_factories = catalogue.create(
        "recon", "operation_factories", entry_points=True
    )


__all__ = [
    "corrections_from_dict",
    "fix_annotations",
    "rename_labels",
    "strip_annotations",
    "split_sentences",
    "operation",
    "Operation",
    "add_tokens",
    "fix_tokenization_and_spacing",
    "filter_overlaps",
    "remove_overlapping_entities",
    "select_subset_of_overlapping_chain",
    "upcase_labels",
]
