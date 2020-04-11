from pathlib import Path
from typing import Any, Dict, List, Set

from spacy.util import ensure_path
import srsly

from .hashing import text_hash
from .types import Example


class ExampleStore:
    def __init__(self, examples: List[Example] = None):
        self._map: Dict[int, Example] = {}
        self._texts: Set[str] = set()
        if examples is not None:
            for e in examples:
                self.add(e)

    def __getitem__(self, example_hash: int):
        return self._map[example_hash]
    
    def add(self, example: Example):
        example_hash = hash(example)
        self._map[example_hash] = example
        self._texts.add(text_hash(example.text))
    
    def __len__(self):
        """The number of strings in the store.
        RETURNS (int): The number of strings in the store.
        """
        return len(self._map)

    def __contains__(self, example):
        """Check whether a string is in the store.
        string (unicode): The string to check.
        RETURNS (bool): Whether the store contains the string.
        """
        return hash(example) in self._map
    
    def contains(self, example):
        return example in self
    
    def contains_text(self, example):
        return example.text in self._texts

    def from_disk(self, path: Path) -> "ExampleStore":
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

        path = ensure_path(path)
        examples = []
        for example_hash, example in self._map.items():
            examples.append({"example_hash": example_hash, "example": example.dict()})
            
        srsly.write_jsonl(path, examples)
