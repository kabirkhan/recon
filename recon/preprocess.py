from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

import catalogue
import spacy
from spacy.language import Language

from recon.linker import BaseEntityLinker, EntityLinker
from recon.types import Entity, Example


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

        e.g. @preprocessor("recon.some_name.v1")

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


class PreProcessor:
    def __init__(self, name: str, field: str) -> None:
        self._name = name
        self._field = field
        self._cache: Dict[Any, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def field(self) -> str:
        return self._field

    def __call__(self, data: Iterable[Example]) -> Iterable[Any]:
        raise NotImplementedError

    def register(self) -> None:
        registry.preprocessors.register(self.name)(self)


class SpacyPreProcessor(PreProcessor):
    def __init__(
        self, nlp: Optional[Language] = None, name: str = "recon.spacy.v1", field: str = "doc"
    ) -> None:
        super().__init__(name, field)
        self._nlp = nlp

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe("sentencizer")
        return self._nlp

    def __call__(self, data: Iterable[Example]) -> Iterable[Any]:
        unseen_texts = (e.text for i, e in enumerate(data) if hash(e) not in self._cache)
        seen_texts = ((i, e.text) for i, e in enumerate(data) if hash(e) in self._cache)

        docs = list(self.nlp.pipe(unseen_texts))
        for doc in docs:
            self._cache[doc.text] = doc
        for idx, st in seen_texts:
            docs.insert(idx, self._cache[st])

        return docs


class SpanAliasesPreProcessor(PreProcessor):
    def __init__(
        self,
        entities: List[Entity],
        name: str = "recon.span_aliases.v1",
        field: str = "aliases",
        linker: BaseEntityLinker = EntityLinker(),
    ):
        super().__init__(name, field)
        self.entities = entities
        self.ents_to_aliases = defaultdict(list)

        for ent in self.entities:
            if not ent.id:
                ent.id = ent.name
            self.ents_to_aliases[ent.id] = ent.aliases

        self.linker = linker

    def __call__(self, data: Iterable[Example]) -> Iterable[Any]:
        outputs = []

        data = self.linker(data)
        for example in data:
            spans_to_aliases_map = defaultdict(list)
            for span in example.spans:
                if span.kb_id:
                    aliases = self.ents_to_aliases[span.kb_id]
                    spans_to_aliases_map[hash(span)] = aliases

            outputs.append(spans_to_aliases_map)
        return outputs


if "recon.spacy.v1" not in registry.preprocessors:
    registry.preprocessors.register("recon.spacy.v1")(SpacyPreProcessor())
