from pathlib import Path
from typing import Any, Dict, List, Set, Union

import srsly
from spacy.util import ensure_path

from .types import Example


class ExampleStore:
    def __init__(self, examples: List[Example] = None):
        self._map: Dict[int, Example] = {}
        if examples is not None:
            for e in examples:
                self.add(e)

    def __getitem__(self, example_hash: int) -> Example:
        return self._map[example_hash]

    def __len__(self) -> int:
        """The number of strings in the store.

        Returns:
            Number of examples in store
        """
        return len(self._map)

    def __contains__(self, example: Union[int, Example]) -> bool:
        """Check whether a string is in the store.
        
        Args:
            example (Union[int, Example]): The example to check

        Returns:
            Whether the store contains the example.
        """
        example_hash = hash(example) if isinstance(example, Example) else example
        return example_hash in self._map

    def add(self, example: Example) -> None:
        """Add an Example to the store
        
        Args:
            example (Example): example to add
        """
        example_hash = hash(example)
        self._map[example_hash] = example

    def from_disk(self, path: Path) -> "ExampleStore":
        """Load store from disk

        Args:
            path (Path): Path to file to load from
        
        Returns:
            ExampleStore: Initialized ExampleStore
        """
        path = ensure_path(path)
        examples = srsly.read_jsonl(path)
        for e in examples:
            example_hash = e["example_hash"]
            raw_example = e["example"]
            example = Example(**raw_example)
            assert hash(example) == example_hash
            self.add(example)

        return self

    def to_disk(self, path: Path) -> None:
        """Save store to disk
        
        Args:
            path (Path): Path to save store to
        """
        path = ensure_path(path)
        examples = []
        for example_hash, example in self._map.items():
            examples.append({"example_hash": example_hash, "example": example.dict()})

        srsly.write_jsonl(path, examples)
