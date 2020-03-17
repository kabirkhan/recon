from functools import reduce
from typing import Any


def compose(*funcs) -> Any:
    """Compose a chain of functions
    TODO: mypy plugin to validate input/output types

    Returns:
        Any: Output of final function in chain
    """    
    return lambda x: reduce(lambda f, g: g(f), list(funcs), x)
