from typing import Any, Dict, List
from pydantic import validator, BaseModel, Field, Schema
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool


class TextSpanLabel(BaseModel):
    text: StrictStr
    start: StrictInt
    end: StrictInt
    label: StrictStr


class Example(BaseModel):
    text: StrictStr
    spans: List[TextSpanLabel]
    meta: Dict[str, Any] = {}


class PredictionError(BaseModel):
    text: StrictStr
    true_label: StrictStr
    false_label: StrictStr
    count: StrictInt
