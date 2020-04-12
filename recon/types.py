from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from pydantic import BaseModel, Field, Schema, root_validator
from .hashing import (
    example_hash,
    span_hash,
    token_hash,
    tokenized_example_hash,
    transformation_hash,
)


class Span(BaseModel):
    """Entity Span in Example"""

    text: str
    start: int
    end: int
    label: str
    token_start: Optional[int]
    token_end: Optional[int]

    def __hash__(self) -> int:
        return cast(int, span_hash(self))


class Token(BaseModel):
    """Token with offsets into Example Text"""

    text: str
    start: int
    end: int
    id: int

    def __hash__(self) -> int:
        return cast(int, token_hash(self))


class Example(BaseModel):
    """Example with NER Label spans"""

    text: str
    spans: List[Span]
    tokens: Optional[List[Token]]
    meta: Dict[str, Any] = {}
    formatted: bool = False

    @root_validator(pre=True)
    def span_text_must_exist(cls, values):
        if not values.get("formatted", False):
            # Ensure each span has a text property
            spans = values["spans"]
            for span in spans:
                if "text" not in span:
                    span["text"] = values["text"][span["start"] : span["end"]]

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


class TransformationType(str, Enum):
    EXAMPLE_ADDED = "EXAMPLE_ADDED"
    EXAMPLE_REMOVED = "EXAMPLE_REMOVED"
    EXAMPLE_CHANGED = "EXAMPLE_CHANGED"


class Transformation(BaseModel):
    prev_example: Optional[int] = None
    example: Optional[int] = None
    type: TransformationType

    def __hash__(self) -> int:
        return cast(int, transformation_hash(self))


class TransformationCallbacks(BaseModel):
    add_example: Callable[[Example], Transformation]
    remove_example: Callable[[int], Transformation]
    change_example: Callable[[int, Example], Transformation]
    track_example: Callable[[Optional[int], Optional[Example]], Transformation]


class OperationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class OperationState(BaseModel):
    name: str
    batch: bool = False
    args: List[Any] = []
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


class OperationResult(BaseModel):
    data: Any
    state: OperationState


class CorpusApplyResult(BaseModel):
    train: Any
    dev: Any
    test: Any
    all: Any


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


class HardestExample(BaseModel):
    """Container for how hard an Example is for an EntityRecognizer to predict
    all entities correctly for.
    """

    example: Example
    count: int
    prediction_errors: Optional[List[PredictionError]]


class LabelDisparity(BaseModel):
    """Container for the number of disparities in a Dataset
    where some text is tagged as label1 in some places and label2 in others
    
    Attributes:
        label1 (str): Label1
        label2 (str): Label2
        count (int): Number of times this label disparity occurs
        examples (List[Example], optional): List of Examples where this disparity occurs
    """

    label1: str
    label2: str
    count: int
    examples: Optional[List[Example]] = []


class NERStats(BaseModel):
    """Container for tracking basic NER statistics"""

    n_examples: int
    n_examples_no_entities: int
    n_annotations: int
    n_annotations_per_type: Dict[str, int]
    examples_with_type: Optional[Dict[str, Example]]


class EntityCoverage(BaseModel):
    """Container for tracking how well an Entity is covered.
    
    Attributes:
        text (str): The entity text
        label (str): The entity label
        count (int): Number of times this text/label combination occurs
        examples (List[Example], optional): List of Examples where this entity occurs
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
