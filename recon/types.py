import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)
from typing_extensions import ParamSpec

from pydantic import BaseModel, field_validator, model_validator
from spacy import displacy
from spacy.tokens import Doc
from spacy.util import get_words_and_spaces
from spacy.vocab import Vocab
from wasabi import color

from recon.hashing import example_hash, prediction_error_hash, span_hash, token_hash

if TYPE_CHECKING:
    from pydantic.typing import ReprArgs

_OpParams = ParamSpec("_OpParams")


ANSI_LABEL = 141
ANSI_HIGHLIGHT = 222
ANSI_BLACK = "black"


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
        return self.hash

    @property
    def hash(self) -> int:
        return span_hash(self)


class Token(BaseModel):
    """Token with offsets into Example Text"""

    text: str
    start: int
    end: int
    id: int

    def __hash__(self) -> int:
        return self.hash

    @property
    def hash(self) -> int:
        return token_hash(self)


class Example(BaseModel):
    """Example with NER Label spans"""

    text: str
    spans: List[Span]
    tokens: Optional[List[Token]] = None
    meta: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def span_text_exists(cls, data: Any) -> Any:
        spans = data.get("spans", [])
        for span in spans:
            if not isinstance(span, dict):
                continue
            if "text" not in span:
                start, end = span["start"], span["end"]
                span["text"] = data["text"][start:end]
        data["spans"] = spans
        return data

    @field_validator("meta", mode="before")
    def validate_meta(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure the meta has a source property
        # if something that's not a dict is passed in
        if isinstance(v, list) or isinstance(v, str):
            return {"source": v}
        return v

    def __str__(self) -> str:
        return f'Example: "{self.text}", {len(self.spans)} spans.'

    def __repr__(self) -> str:
        n_spans = len(self.spans)
        spans_text = "span" if n_spans == 1 else "spans"
        return f'Example: "{self.text}", {n_spans} {spans_text}.'

    def __hash__(self) -> int:
        return self.hash

    @property
    def hash(self) -> int:
        return example_hash(self)

    @property
    def doc(self) -> Doc:
        """Return spaCy Doc representation of Example

        Returns:
            Doc: Output spaCy Doc with ents set from example spans
        """
        if not self.tokens:
            raise ValueError(
                "Tokens are not set. Try running the recon.add_tokens.v1 operation."
            )
        tokens = [token.text for token in self.tokens]
        words, spaces = get_words_and_spaces(tokens, self.text)
        doc = Doc(Vocab(), words=words, spaces=spaces)
        doc.ents = tuple(
            doc.char_span(s.start, s.end, label=s.label) for s in self.spans
        )
        return doc

    def show(
        self,
        jupyter: Optional[bool] = None,
        options: Dict[str, Any] = {},
        highlight_color: Union[str, int] = ANSI_HIGHLIGHT,
        label_color: Union[str, int] = ANSI_LABEL,
    ) -> None:
        """Visualize example using spaCy displacy entity renderer or a simple renderer
        for console output.

        Args:
            jupyter (Optional[bool], optional): Run for Jupyter Interactive Environment.
            options (Dict[str, Any]): DisplaCy options to
                pass through to filter ent types and set colors
                See: https://spacy.io/usage/visualizers#ent
        """
        if sys.stdin.isatty():
            self.pretty_print(highlight_color, label_color)
        else:
            displacy.render(self.doc, style="ent", jupyter=jupyter, options=options)

    def pretty_print(
        self,
        highlight_color: Union[str, int] = ANSI_HIGHLIGHT,
        label_color: Union[str, int] = ANSI_LABEL,
    ) -> None:
        """Pretty print an Example's spans for console output.

        Args:
            highlight_color (Union[str, int], optional): ANSI color code or name.
                Defaults to 222 (yellowish)
            label_color (Union[str, int], optional): ANSI color code or name.
                Defaults to 141 (purpleish)
        """
        text = self.text
        spans = self.spans
        result = ""
        offset = 0
        for span in spans:
            label = span.label
            start = span.start
            end = span.end
            result += text[offset:start]
            result += color(f" {text[start:end]} ", ANSI_BLACK, highlight_color)
            if label:
                result += color(f" {label} ", ANSI_BLACK, label_color)
            offset = end
        result += text[offset:]
        print(result)


class OperationProtocol(Protocol[_OpParams]):
    def __call__(
        self, example: Example, *args: _OpParams.args, **kwargs: _OpParams.kwargs
    ) -> Union[Example, Iterable[Example], None]:
        ...


class StatsProtocol(Protocol[_OpParams]):
    def __call__(
        self, data: List[Example], *args: _OpParams.args, **kwargs: _OpParams.kwargs
    ) -> Any:
        ...


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


def add_shim(example: Example) -> None:
    return None


def remove_shim(example_hash: int) -> None:
    return None


def change_shim(example_hash: int, new_example: Example) -> None:
    return None


def track_shim(
    example_hash: Optional[int] = None, new_example: Optional[Example] = None
) -> None:
    return None


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
    commit: int
    size: int
    operations: List[OperationState]


# class DatasetMeta(BaseModel):
#     name: str
#     version: str


class CorpusMeta(BaseModel):
    name: str
    # versions: List[DatasetMeta]


class OperationResult(BaseModel):
    data: List[Example]
    state: OperationState


class CorpusApplyResult(BaseModel):
    train: Any
    dev: Any
    test: Any
    all: Any

    def items(self) -> List[Tuple[str, Any]]:
        return [
            ("train", self.train),
            ("dev", self.dev),
            ("test", self.test),
            ("all", self.all),
        ]


class AnnotationCount(BaseModel):
    text: str
    count: int
    examples: List[Example]

    def __repr_args__(self) -> 'ReprArgs':
        return [arg for arg in super().__repr_args__() if arg[0] != "examples"]


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
        return self.hash

    @property
    def hash(self) -> int:
        return prediction_error_hash(self)

    def __repr_args__(self) -> 'ReprArgs':
        return [arg for arg in super().__repr_args__() if arg[0] != "examples"]


class ExampleDiff(BaseModel):
    reference: Example
    prediction: Example
    count: int
    score: float

    def show(self, label_suffix: str = "PRED"):
        combined = self.reference.copy(deep=True)
        pred_spans = []
        for s in self.prediction.spans:
            span = s.model_copy(deep=True)
            span.label = f"{s.label}:{label_suffix}"
            pred_spans.append(span)
        combined.spans = sorted(combined.spans + pred_spans, key=lambda s: s.start)
        assert combined.tokens is not None
        tokens = [token.text for token in combined.tokens]
        words, spaces = get_words_and_spaces(tokens, combined.text)
        doc = Doc(Vocab(), words=words, spaces=spaces)
        doc.spans["ref"] = [
            doc.char_span(s.start, s.end, label=s.label) for s in combined.spans
        ]
        displacy.render(doc, style="span", jupyter=True, options={"spans_key": "ref"})


HardestExample = ExampleDiff


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

    def __repr_args__(self) -> 'ReprArgs':
        return [arg for arg in super().__repr_args__() if arg[0] != "examples"]


class EntityCoverageStats(BaseModel):
    """Container for output of how similar the Entity Coverage of 2 datasets is"""

    entity: float
    count: float


class Outliers(BaseModel):
    """Container for low and high indices of outlier detection"""

    low: List[int]
    high: List[int]


class Correction(BaseModel):
    """Container for an annotation correction, mapping an
    annotation from a label to a label
    """

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

                will return a list of Corrections with 1 item,
                correcting the text "model" from PRODUCT
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
                    "Cannot parse corrections dict. Value must be either a str of the"
                    " label "
                    + "to change the annotation to (TO_LABEL) or a tuple of"
                    " (FROM_LABEL, TO_LABEL)"
                )
            corrections.append(
                cls(annotation=key, from_labels=from_labels, to_label=to_label)
            )
        return corrections


class Scores(BaseModel):
    ents_p: float
    ents_r: float
    ents_f: float
    ents_per_type: Dict[str, Any]
    speed: float
