from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from spacy.kb import KnowledgeBase

from recon.types import Example


class BaseEntityLinker(ABC):
    @abstractmethod
    def __call__(self, examples: Iterable[Example]) -> Iterator[Example]:
        raise NotImplementedError


class EntityLinker(BaseEntityLinker):
    def __call__(self, examples: Iterable[Example]) -> Iterator[Example]:
        for example in examples:
            for span in example.spans:
                span.kb_id = span.text
            yield example


class SpacyEntityLinker(BaseEntityLinker):
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def __call__(self, examples: Iterable[Example]) -> Iterator[Example]:
        for example in examples:
            for span in example.spans:
                cands = self.kb.get_candidates(span.text)
                ents = [c.entity_ for c in cands if c.entity_]
                if ents:
                    top_ent = ents[0]
                    span.kb_id = top_ent

            yield example
