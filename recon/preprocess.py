from typing import Any, Callable, Dict, Iterable, List

import catalogue
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .types import Example


class registry:
    preprocessors = catalogue.create("recon", "preprocessors", entry_points=True)


class preprocessor:
    def __init__(self, name: str):
        """Decorate an operation that makes some changes to a dataset.

        Args:
            name (str): Operation name.
        """
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """Decorator for an operation.
        The first arg is the function being decorated.
        This function operates on a List[Example].

        e.g. @preprocessor("recon.v1.some_name")

        Or it should operate on a single example and
        recon will take care of applying it to a full Dataset

        Args:
            args: First arg is function to decorate

        Returns:
            Callable: Original function
        """
        op: Callable = args[0]
        registry.preprocessors.register(self.name)(op)

        return op


class PreProcessor(object):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._cache: Dict[Any, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, data: List[Example]) -> Iterable[Any]:
        raise NotImplementedError

    def register(self) -> None:
        registry.preprocessors.register(self.name)(self)


class SpacyPreProcessor(PreProcessor):
    def __init__(self, name: str = "recon.v1.spacy", nlp: Language = None) -> None:
        super().__init__(name)
        self._name = name
        self._nlp = nlp

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe(self._nlp.create_pipe("sentencizer"))
        return self._nlp

    def __call__(self, data: List[Example]) -> Iterable[Any]:
        unseen_texts = (e.text for i, e in enumerate(data) if hash(e) not in self._cache)
        seen_texts = ((i, e.text) for i, e in enumerate(data) if hash(e) in self._cache)

        docs = list(self.nlp.pipe(unseen_texts))
        for doc in docs:
            self._cache[doc.text] = doc
        for idx, st in seen_texts:
            docs.insert(idx, self._cache[st])

        return docs


if "recon.v1.spacy" not in registry.preprocessors:
    registry.preprocessors.register("recon.v1.spacy")(SpacyPreProcessor())
