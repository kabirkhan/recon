from functools import reduce
from typing import Any, Callable, Sequence


def compose(*funcs: Any) -> Any:
    """Compose a chain of functions
    TODO: mypy plugin to validate input/output types

    Returns:
        Any: Output of final function in chain
    """
    return lambda x: reduce(lambda f, g: g(f), list(funcs), x)
