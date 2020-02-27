from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, Schema, validator


class TextSpanLabel(BaseModel):
    text: str
    start: int
    end: int
    label: str


class Example(BaseModel):
    text: str
    spans: List[TextSpanLabel]
    meta: Dict[str, Any] = {}


class PredictionError(BaseModel):
    text: str
    true_label: str
    pred_label: str
    count: int
    examples: Optional[List[Example]] = []
