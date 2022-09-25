import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Set

from spacy.language import Language
from spacy.training.corpus import Corpus as SpacyCorpus
from wasabi import Printer

from recon.loaders import to_spacy
from recon.types import Example, Scores, Span, Token


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

    def _evaluate(self, examples: List[Example]) -> Scores:
        raise NotImplementedError

    def evaluate(self, data: List[Example], verbose: bool = True) -> Scores:
        """Evaluate recognizer performance on dataset and print metrics

        Args:
            data (List[Example]): Examples to evaluate on
            verbose (bool, optional): Print results or not. Defaults to True.

        Returns:
            Scorer: spaCy scorer object
        """
        sc = self._evaluate(data)

        msg = Printer(no_print=not verbose)
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
        msg.table(table_data, header=header, divider=True)
        return sc


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

    def _evaluate(self, data: List[Example]) -> Scores:
        """Evaluate spaCy recognizer performance on dataset

        Args:
            data (List[Example]): Examples to evaluate on
            verbose (bool, optional): Print results or not. Defaults to True.

        Returns:
            Scorer: spaCy scorer object
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "data.spacy"
            to_spacy(data_path, data)
            corpus = SpacyCorpus(data_path, gold_preproc=False)
            dev_dataset = list(corpus(self.nlp))
            sc = self.nlp.evaluate(dev_dataset)
            scores = Scores(**sc)
        return scores
