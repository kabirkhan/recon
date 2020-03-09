from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, Schema, validator


class Span(BaseModel):
    text: str
    start: int
    end: int
    label: str


class Example(BaseModel):
    text: str
    spans: List[Span]
    meta: Dict[str, Any] = {}


class PredictionError(BaseModel):
    text: str
    true_label: str
    pred_label: str
    count: int
    examples: Optional[List[Example]] = []


class LabelDisparity(BaseModel):
    label1: str
    label2: str
    count: int
    examples: Optional[List[Example]] = []


class EntityCoverage(BaseModel):
    text: str
    label: str
    count: int
    examples: Optional[List[Example]] = []


class HardestExample(BaseModel):
    example: Example
    count: int
    prediction_errors: Optional[List[PredictionError]]
