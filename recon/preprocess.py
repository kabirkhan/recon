from typing import Any, Dict, Iterable, List

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .types import Example


class PreProcessor(object):
    name = "recon.v1.preprocess"

    def __init__(self) -> None:
        super().__init__()
        self._cache: Dict[Any, Any] = {}

    def __call__(self, data: List[Example]) -> Iterable[Any]:
        raise NotImplementedError


class SpacyPreProcessor(PreProcessor):
    name = "recon.v1.spacy"

    def __init__(self, nlp: Language) -> None:
        super().__init__()
        self._nlp = nlp

    def __call__(self, data: List[Example]) -> Iterable[Any]:
        unseen_texts = (e.text for i, e in enumerate(data) if hash(e) not in self._cache)
        seen_texts = ((i, e.text) for i, e in enumerate(data) if hash(e) in self._cache)

        docs = list(self._nlp.pipe(unseen_texts))
        for doc in docs:
            self._cache[doc.text] = doc
        for idx, st in seen_texts:
            docs.insert(idx, self._cache[st])

        return docs
