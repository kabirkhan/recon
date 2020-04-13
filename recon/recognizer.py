from typing import Dict, Iterable, Iterator, List, Set, Tuple

import srsly
from spacy.language import Language
from wasabi import Printer

from .types import Example, Span, Token


class EntityRecognizer:
    """Abstract Base class for recognizing entities in a batch of text.
    Used in the `recon.insights` module for understanding the kinds
    of examples your model is having the most trouble with.
    """

    @property
    def labels(self) -> List[str]:
        """Return List of String Labels
        
        Raises:
            NotImplementedError: Not Implemented, override
        
        Returns:
            List[str]: List of labels the model can predict
        """
        raise NotImplementedError

    def predict(self, texts: Iterable[str]) -> Iterator[Example]:
        """Run model inference on a batch of raw texts.
        
        Args:
            texts (Iterable[str]): Raw text examples
        
        Raises:
            NotImplementedError: Not implemented, override
        
        Returns:
            Iterator[Example]: Iterator of Examples
        """
        raise NotImplementedError


class SpacyEntityRecognizer(EntityRecognizer):
    """Create an EntityRecognizer from a spaCy Langauge instance"""

    def __init__(self, nlp: Language):
        """Initialize a SpacyEntityRecognizer
        
        Args:
            nlp (Language): spaCy Language instance that can sets doc.ents
        """
        super().__init__()
        self.nlp = nlp

    @property
    def labels(self) -> List[str]:
        """Return List of spaCy ner labels
        
        Returns:
            List[str]: List of labels from spaCy ner pipe
        """
        all_labels: Set[str] = set()

        for pipe in ["ner", "entity_ruler"]:
            if self.nlp.has_pipe(pipe):
                all_labels = all_labels | set(self.nlp.get_pipe(pipe).labels)

        return sorted(list(all_labels))

    def predict(self, texts: Iterable[str]) -> Iterator[Example]:
        """Run spaCy nlp.pipe on a batch of raw texts.
        
        Args:
            texts (Iterable[str]): Raw text examples
        
        Yields:
            Iterator[Example]: Examples constructed from spaCy Model predictions
        """
        examples: List[Example] = []

        for doc in self.nlp.pipe(texts):
            yield Example(
                text=doc.text,
                spans=[
                    Span(
                        text=e.text,
                        start=e.start_char,
                        end=e.end_char,
                        label=e.label_,
                        token_start=e.start,
                        token_end=e.end,
                    )
                    for e in doc.ents
                ],
                tokens=[Token(text=t.text, start=t.idx, end=t.idx + len(t), id=t.i) for t in doc],
            )

    def evaluate(self, data: List[Example]) -> None:
        msg = Printer()
        formatted_data, _ = self._format_data(data)
        sc = self.nlp.evaluate(formatted_data, batch_size=64)
        msg.divider("Recognizer Results")
        result = [
            ("Precision", f"{sc.ents_p:.3f}"),
            ("Recall", f"{sc.ents_r:.3f}"),
            ("F-Score", f"{sc.ents_f:.3f}"),
        ]
        msg.table(result)

        table_data = []
        for label, scores in sorted(sc.ents_per_type.items(), key=lambda tup: tup[0]):
            table_data.append(
                (label, f"{scores['p']:.3f}", f"{scores['r']:.3f}", f"{scores['f']:.3f}")
            )
        header = ("Label", "Precision", "Recall", "F-Score")
        formatted = msg.table(table_data, header=header, divider=True)
        return sc

    def _format_data(
        self, data: List[Example]
    ) -> Tuple[List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], Set[str]]:
        result = []
        labels = set()
        for example in data:
            ents = [(s.start, s.end, s.label) for s in example.spans]
            labels.update([ent[2] for ent in ents])
            result.append((example.text, {"entities": ents}))
        return result, labels
