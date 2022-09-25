"""Utils for running operations and resolving parameters dynamically. The utilties for resolving parameters dynamically are
based on the functionality from FastAPI's process of resolving route params and request bodies to Pydantic Models"""


import functools
import inspect
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseConfig, BaseModel
from pydantic.class_validators import Validator
from pydantic.error_wrappers import ErrorWrapper
from pydantic.errors import MissingError
from pydantic.fields import (
    SHAPE_LIST,
    SHAPE_SEQUENCE,
    SHAPE_SET,
    SHAPE_SINGLETON,
    SHAPE_TUPLE,
    SHAPE_TUPLE_ELLIPSIS,
    FieldInfo,
    ModelField,
    Required,
    UndefinedType,
)
from pydantic.schema import get_annotation_from_field_info
from pydantic.typing import ForwardRef, evaluate_forwardref
from pydantic.utils import lenient_issubclass

from recon.types import OperationState

sequence_shapes = {
    SHAPE_LIST,
    SHAPE_SET,
    SHAPE_TUPLE,
    SHAPE_SEQUENCE,
    SHAPE_TUPLE_ELLIPSIS,
}
sequence_types = (list, set, tuple)
sequence_shape_to_type = {
    SHAPE_LIST: list,
    SHAPE_SET: set,
    SHAPE_TUPLE: tuple,
    SHAPE_SEQUENCE: list,
    SHAPE_TUPLE_ELLIPSIS: list,
}


def is_scalar_field(field: ModelField) -> bool:
    field.field_info
    if not (
        field.shape == SHAPE_SINGLETON
        and not lenient_issubclass(field.type_, BaseModel)
        and not lenient_issubclass(field.type_, sequence_types + (dict,))
    ):
        return False
    if field.sub_fields:
        if not all(is_scalar_field(f) for f in field.sub_fields):
            return False
    return True


def is_scalar_sequence_field(field: ModelField) -> bool:
    if (field.shape in sequence_shapes) and not lenient_issubclass(field.type_, BaseModel):
        if field.sub_fields is not None:
            for sub_field in field.sub_fields:
                if not is_scalar_field(sub_field):
                    return False
        return True
    if lenient_issubclass(field.type_, sequence_types):
        return True
    return False


def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param, globalns),
        )
        for param in signature.parameters.values()
    ]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature


def get_typed_annotation(param: inspect.Parameter, globalns: Dict[str, Any]) -> Any:
    annotation = param.annotation
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation


def create_response_field(
    name: str,
    type_: Type[Any],
    class_validators: Optional[Dict[str, Validator]] = None,
    default: Optional[Any] = None,
    required: Union[bool, UndefinedType] = False,
    model_config: Type[BaseConfig] = BaseConfig,
    field_info: Optional[FieldInfo] = None,
    alias: Optional[str] = None,
) -> ModelField:
    """
    Create a new response field. Raises if type_ is invalid.
    """
    class_validators = class_validators or {}
    field_info = field_info or FieldInfo(None)

    response_field = functools.partial(
        ModelField,
        name=name,
        type_=type_,
        class_validators=class_validators,
        default=default,
        required=required,
        model_config=model_config,
        alias=alias,
    )

    try:
        return response_field(field_info=field_info)
    except RuntimeError:
        raise ValueError(
            f"Invalid args for response field! Hint: check that {type_} is a valid pydantic field type"
        )


def get_param_field(
    *,
    param: inspect.Parameter,
    param_name: str,
    default_field_info: Type[FieldInfo] = FieldInfo,
    ignore_default: bool = False,
) -> ModelField:
    default_value = Required
    had_schema = False
    if not param.default == param.empty and ignore_default is False:
        default_value = param.default
    if isinstance(default_value, FieldInfo):
        had_schema = True
        field_info = default_value
        default_value = field_info.default
    else:
        field_info = default_field_info(default_value)
    required = default_value == Required
    annotation: Any = Any
    if not param.annotation == param.empty:
        annotation = param.annotation
    annotation = get_annotation_from_field_info(annotation, field_info, param_name)
    if not field_info.alias and getattr(field_info, "convert_underscores", None):
        alias = param.name.replace("_", "-")
    else:
        alias = field_info.alias or param.name
    field = create_response_field(
        name=param.name,
        type_=annotation,
        default=None if required else default_value,
        alias=alias,
        required=required,
        field_info=field_info,
    )
    field.required = required
    if not had_schema and not is_scalar_field(field=field):
        field.field_info = field_info.default

    return field


def request_body_to_args(
    required_params: List[ModelField],
    received_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[ErrorWrapper]]:
    values = {}
    errors = []
    if required_params:
        field = required_params[0]
        field_info = field.field_info
        embed = getattr(field_info, "embed", None)
        field_alias_omitted = len(required_params) == 1 and not embed
        if field_alias_omitted:
            received_body = {field.alias: received_body}

        for field in required_params:

            loc: Tuple[str, ...]
            if field_alias_omitted:
                loc = ("body",)
            else:
                loc = ("body", field.alias)

            value = None
            if received_body is not None:
                try:
                    value = received_body.get(field.alias)
                except AttributeError:
                    errors.append(get_missing_field_error(loc))
                    continue
            if value is None:
                if field.required:
                    errors.append(get_missing_field_error(loc))
                else:
                    values[field.name] = deepcopy(field.default)
                continue

            v_, errors_ = field.validate(value, values, loc=loc)

            if isinstance(errors_, ErrorWrapper):
                errors.append(errors_)
            elif isinstance(errors_, list):
                errors.extend(errors_)
            else:
                values[field.name] = v_
    return values, errors


def get_missing_field_error(loc: Tuple[str, ...]) -> ErrorWrapper:
    missing_field_error = ErrorWrapper(MissingError(), loc=loc)
    return missing_field_error


def get_required_operation_params(op: Callable) -> Dict[str, ModelField]:
    """Get required typed parameters for an operation

    Based on logic for JSON resolution to Pydantic types implemented by FastAPI
    Reference: https://github.com/tiangolo/fastapi/blob/master/fastapi/dependencies/utils.py

    Args:
        op (Callable): Inner Callable of a recon operation

    Returns:
        Dict[str, ModelField]: Mapping of field name to Pydantic ModelField for
            required parameters of the callable
    """

    endpoint_signature = get_typed_signature(op)
    signature_params = endpoint_signature.parameters

    required_params = OrderedDict()
    for param_name, param in signature_params.items():
        if param_name in {"example", "preprocessed_outputs"}:
            # All operations accept an example and potentially a special
            # preprocessed_outputs parameter if the operation requires
            # batched example preprocessing to function. We don't
            # need to include these in dynamic param resolution as they are
            # already handled by the internals of the operation
            continue

        param_field = get_param_field(
            param=param, default_field_info=FieldInfo, param_name=param_name
        )
        required_params[param_name] = param_field

    return required_params


def get_received_operation_data(
    required_params: Dict[str, ModelField], state: OperationState
) -> Dict[str, Any]:
    """Resolve serialized args and kwargs data of an operation to their Pydantic types

    Args:
        required_params (Dict[str, ModelField]): Mapping of field name to Pydantic ModelField for
            required parameters of the callable
        state (OperationState): Operation state including serialized args and kwargs data

    Returns:
        Dict[str, Any]: Kwargs mapping of deserialized params to pass to the operation
    """

    filled_keys = set()
    received_data = {}

    if state.kwargs:
        for key, val in state.kwargs.items():
            if key in required_params:
                received_data[key] = val
                filled_keys.add(key)

    if set(required_params.keys()) - filled_keys and state.args:
        for arg_idx, arg in enumerate(state.args):
            param_field = required_params[list(required_params.keys())[arg_idx]]
            received_data[param_field.name] = arg

    return received_data
