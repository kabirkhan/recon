"""Load a list of Example data from JSON represented records in
the [Prodigy](https://prodi.gy) format.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import spacy
import srsly
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces

from recon.types import Example, Span, Token


def read_jsonl(path: Path) -> List[Example]:
    """Read annotations in JSONL file format

    Args:
        path (Path): Path to data

    Returns:
        List[Example]: List of examples
    """
    data = srsly.read_jsonl(path)
    examples = json_to_examples(data)
    return examples


def read_json(path: Path) -> List[Example]:
    """Read annotations in JSON file format

    Args:
        path (Path): Path to data

    Returns:
        List[Example]: List of examples
    """
    data = srsly.read_json(path)
    examples = json_to_examples(data)
    return examples


def json_to_examples(data: List[Dict[str, Any]]) -> List[Example]:
    """Convert List of Dicts to List of typed Examples

    Args:
        data (List[Dict[str, Any]]): Input List of Dicts to convert

    Returns:
        List[Example]: List of typed Examples
    """
    return [Example(**example) for example in data]


def from_spacy(
    path: Path, nlp: Optional[Language] = None, lang_code: str = "en"
) -> Iterable[Example]:
    """Load examples from .spacy docbin format

    Args:
        path (Path): Path to data
        nlp (Language, optional): Spacy Language object.
        lang_code (str, optional): Language code to create a blank spacy model with if nlp is not provided.

    Yields:
        Iterable[Example]: List of typed Examples
    """
    if not nlp:
        nlp = spacy.blank(lang_code)

    doc_bin = DocBin().from_disk(path)
    for doc in doc_bin.get_docs(nlp.vocab):
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


def to_spacy(
    path: Path, data: Iterable[Example], nlp: Optional[Language] = None, lang_code: str = "en"
) -> DocBin:
    """Save a batch of examples to disk in the .spacy DocBin format

    Args:
        path (Path): Path to data
        data (Iterable[Example]): Input Examples
        nlp (Language, optional): Spacy Language object.
        lang_code (str, optional): Language code to create a blank spacy model with if nlp is not provided.

    Returns:
        DocBin: Spacy DocBin with stored example data.
    """

    if not nlp:
        nlp = spacy.blank(lang_code)

    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for example in data:
        if example.tokens:
            tokens = [token.text for token in example.tokens]
            words, spaces = get_words_and_spaces(tokens, example.text)
            doc = Doc(nlp.vocab, words=words, spaces=spaces)
            doc.ents = tuple([doc.char_span(s.start, s.end, label=s.label) for s in example.spans])
            doc_bin.add(doc)
    doc_bin.to_disk(path)
    return doc_bin
