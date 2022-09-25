from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from pydantic import BaseModel, Extra, root_validator
from spacy import displacy
from spacy.tokens import Doc
from spacy.util import get_words_and_spaces
from spacy.vocab import Vocab

from recon.hashing import (
    prediction_error_hash,
    span_hash,
    token_hash,
    tokenized_example_hash,
)


class Span(BaseModel):
    """Entity Span in Example"""

    text: str
    start: int
    end: int
    label: str
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    kb_id: Optional[str] = None
    source: Optional[str] = None

    def __hash__(self) -> int:
        return cast(int, span_hash(self))

    @property
    def hash(self) -> str:
        return cast(str, span_hash(self, as_int=False))


class Token(BaseModel):
    """Token with offsets into Example Text"""

    text: str
    start: int
    end: int
    id: int

    def __hash__(self) -> int:
        return cast(int, token_hash(self))

    @property
    def hash(self) -> str:
        return cast(str, token_hash(self, as_int=False))


class Example(BaseModel):
    """Example with NER Label spans"""

    text: str
    spans: List[Span]
    tokens: Optional[List[Token]]
    meta: Dict[str, Any] = {}
    formatted: bool = False

    class Config:
        extra = Extra.allow

    @root_validator(pre=True)
    def span_text_must_exist(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("formatted", False):
            # Ensure each span has a text property
            spans = values["spans"]
            for span in spans:
                if not isinstance(span, Span):
                    if "text" not in span:
                        start, end = span["start"], span["end"]
                        span["text"] = values["text"][start:end]

            # Ensure the meta has a source property
            # if something that's not a dict is passed in
            meta = values.get("meta", {})
            if isinstance(meta, list) or isinstance(meta, str):
                meta = {"source": meta}

            values["spans"] = spans
            values["meta"] = meta
            values["formatted"] = True

        return values

    def __hash__(self) -> int:
        return cast(int, tokenized_example_hash(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Example):
            return self.dict() == other.dict()
        return False

    def dict(self, **kwargs: Any) -> Dict:
        res = super().dict(**kwargs)
        keys = list(res.keys())
        for k in keys:
            if k not in self.schema()["properties"].keys():
                del res[k]
        return res

    @property
    def hash(self) -> str:
        return cast(str, tokenized_example_hash(self, as_int=False))

    @property
    def doc(self) -> Doc:
        """Return spaCy Doc representation of Example

        Returns:
            Doc: Output spaCy Doc with ents set from example spans
        """
        if not self.tokens:
            raise ValueError("Tokens are not set. Try running the recon.add_tokens.v1 operation.")
        tokens = [token.text for token in self.tokens]
        words, spaces = get_words_and_spaces(tokens, self.text)
        doc = Doc(Vocab(), words=words, spaces=spaces)
        doc.ents = tuple(doc.char_span(s.start, s.end, label=s.label) for s in self.spans)
        return doc

    def show(self, jupyter: Optional[bool] = None, options: Dict[str, Any] = {}) -> None:
        """Visualize example using spaCy displacy entity renderer

        Args:
            jupyter (Optional[bool], optional): Run for Jupyter Interactive Environment.
            options (Dict[str, Any]): DisplaCy options to pass through to filter ent types and set colors
                See: https://spacy.io/usage/visualizers#ent
        """
        displacy.render(self.doc, style="ent", jupyter=jupyter, options=options)


OpType = Callable[[Example, Any], Any]
BatchOpType = Callable[[List[Example], Any], Any]
ApplyType = Union[str, BatchOpType]
ApplyInPlaceType = Union[str, BatchOpType]


class Entity(BaseModel):
    id: Optional[str] = None
    name: str
    aliases: List[str]


class TransformationType(str, Enum):
    EXAMPLE_ADDED = "EXAMPLE_ADDED"
    EXAMPLE_REMOVED = "EXAMPLE_REMOVED"
    EXAMPLE_CHANGED = "EXAMPLE_CHANGED"


class Transformation(BaseModel):
    prev_example: Optional[int] = None
    example: Optional[int] = None
    type: TransformationType


# fmt: off
def add_shim(example: Example) -> None:
    return None


def remove_shim(example_hash: int) -> None:
    return None


def change_shim(example_hash: int, new_example: Example) -> None:
    return None


def track_shim(example_hash: Optional[int] = None, new_example: Optional[Example] = None) -> None:
    return None
# fmt: on


@dataclass
class TransformationCallbacks:
    add_example: Callable[[Example], None] = add_shim
    remove_example: Callable[[int], None] = remove_shim
    change_example: Callable[[int, Example], None] = change_shim
    track_example: Callable[[Optional[int], Optional[Example]], None] = track_shim


class OperationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    NEEDS_TOKENIZATION = "NEEDS_TOKENIZATION"


class OperationState(BaseModel):
    name: str
    batch: bool = False
    args: Tuple[Any, ...] = tuple()
    kwargs: Dict[str, Any] = {}
    status: OperationStatus = OperationStatus.NOT_STARTED
    ts: datetime = datetime.now()
    examples_added: int = 0
    examples_removed: int = 0
    examples_changed: int = 0
    transformations: List[Transformation] = []


class DatasetOperationsState(BaseModel):
    name: str
    commit: str
    size: int
    operations: List[OperationState]


# class DatasetMeta(BaseModel):
#     name: str
#     version: str


class CorpusMeta(BaseModel):
    name: str
    # versions: List[DatasetMeta]


class OperationResult(BaseModel):
    data: Any
    state: OperationState


class CorpusApplyResult(BaseModel):
    train: Any
    dev: Any
    test: Any
    all: Any

    def items(self) -> List[Tuple[str, Any]]:
        return [("train", self.train), ("dev", self.dev), ("test", self.test), ("all", self.all)]


class AnnotationCount(BaseModel):
    text: str
    count: int
    examples: List[Example]


class PredictionErrorExamplePair(BaseModel):
    """Dataclass representation of original Example in a PredictionError
    vs the predicted Example from an EntityRecognizer

    Attributes:
        original (Example): Original Example
        spans (List[Span]): List of entity spans
        meta (Dict[str, Any], optional): Meta information about the example
    """

    original: Example
    predicted: Example


class PredictionError(BaseModel):
    """Representation of errors an EntityRecognizer makes on a labeled dataset.

    Attributes:
        text (str): Span text with error
        true_label (str): True label in annotated Example
        pred_label (str): The label predicted by the EntityRecognizer
        count (int): Number of times this PredictionError occurs
        examples (List[PredictionErrorExamplePair], optional):
            List of PredictionErrorExamplePairs that have this PredictionError
    """

    text: str
    true_label: str
    pred_label: str
    count: int
    examples: Optional[List[PredictionErrorExamplePair]] = []

    def __hash__(self) -> int:
        return cast(int, prediction_error_hash(self))

    @property
    def hash(self) -> str:
        return cast(str, prediction_error_hash(self, as_int=False))


class HardestExample(BaseModel):
    reference: Example
    prediction: Example
    count: int
    score: float


class LabelDisparity(BaseModel):
    """Container for the number of disparities in a Dataset
    where some text is tagged as label1 in some places and label2 in others

    Attributes:
        label1 (str): Label1
        label2 (str): Label2
        count (int): Number of times this label disparity occurs
        examples (List[Example], optional): List of examples where this disparity occurs
    """

    label1: str
    label2: str
    count: int
    examples: Optional[List[Example]] = []


class Stats(BaseModel):
    """Container for tracking basic NER statistics"""

    n_examples: int
    n_examples_no_entities: int
    n_annotations: int
    n_annotations_per_type: Dict[str, int]
    examples_with_type: Optional[Dict[str, Example]] = None

    def __str__(self) -> str:
        return self.json(indent=4, exclude_unset=True)


class EntityCoverage(BaseModel):
    """Container for tracking how well an Entity is covered.

    Attributes:
        text (str): The entity text
        label (str): The entity label
        count (int): Number of times this text/label combination occurs
        examples (List[Example], optional): List of examples where this entity occurs
    """

    text: str
    label: str
    count: int
    examples: Optional[List[Example]] = []

    def __hash__(self) -> int:
        return hash((self.text, self.label))


class EntityCoverageStats(BaseModel):
    """Container for output of how similar the Entity Coverage of 2 datasets is"""

    entity: float
    count: float


class Outliers(BaseModel):
    """Container for low and high indices of outlier detection"""

    low: List[int]
    high: List[int]


class Correction(BaseModel):
    """Container for an annotation correction, mapping an annotation from a label to a label"""

    annotation: str
    from_labels: List[str]
    to_label: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "List[Correction]":
        """Load a list of Corrections from the Dict short-hand syntax.

        Args:
            obj (Dict[str, Any]): Dictionary of corrections.
                Example:

                corrections = Correction.from_dict({
                    "model": ("PRODUCT", "SKILL"),
                })

                will return a list of Corrections with 1 item, correcting the text "model" from PRODUCT
                to SKILL.

        Raises:
            ValueError: If Dict format is invalid

        Returns:
            List[Correction]: List of Corrections
        """
        corrections: List[Correction] = []
        for key, val in obj.items():
            if isinstance(val, str) or val is None:
                from_labels = ["ANY"]
                to_label = val
            elif isinstance(val, tuple):
                if isinstance(val[0], str):
                    from_labels = [val[0]]
                else:
                    from_labels = val[0]
                to_label = val[1]
            else:
                raise ValueError(
                    "Cannot parse corrections dict. Value must be either a str of the label "
                    + "to change the annotation to (TO_LABEL) or a tuple of (FROM_LABEL, TO_LABEL)"
                )
            corrections.append(cls(annotation=key, from_labels=from_labels, to_label=to_label))
        return corrections


class Scores(BaseModel):
    ents_p: float
    ents_r: float
    ents_f: float
    ents_per_type: Dict[str, Any]
    speed: float
