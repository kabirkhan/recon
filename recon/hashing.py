import hashlib
from typing import Any, Callable, Tuple, Union


def token_hash(token: Any, as_int: bool = True) -> Union[str, int]:
    """Hash of Token type
    
    Args:
        token (Token): Token to hash
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Token hash
    """
    return _hash((token.text, token.start, token.end, token.id), as_int=as_int)


def span_hash(span: Any, as_int: bool = True) -> Union[str, int]:
    """Hash of Span type
    
    Args:
        span (Span): Span to hash
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Span hash
    """
    hash_data = (
        span.start,
        span.end,
        span.label,
        span.text,
        span.token_start if span.token_start else 0,
        span.token_end if span.token_end else 0,
    )
    return _hash(hash_data, as_int=as_int)


def example_hash(example: Any, as_int: bool = True) -> Union[str, int]:
    """Hash of Example type
    
    Args:
        example (Example): Example to hash
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Example hash
    """
    hash_data = (example.text,) + tuple((span_hash(span, as_int=False) for span in example.spans))
    return _hash(hash_data, as_int=as_int)


def tokenized_example_hash(example: Any, as_int: bool = True) -> Union[str, int]:
    """Hash of Example type including token data
    
    Args:
        example (Example): Example to hash
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Example hash
    """
    tokens = example.tokens or []
    hash_data = (
        (example.text,)
        + tuple((span_hash(span, as_int=False) for span in example.spans))
        + tuple((token_hash(token, as_int=False) for token in tokens))
    )
    return _hash(hash_data, as_int=as_int)


def dataset_hash(dataset: Any, as_int: bool = True) -> Union[str, int]:
    """Hash of Dataset
    
    Args:
        dataset (Dataset): Dataset to hash
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Dataset hash
    """
    hash_data = (dataset.name,) + tuple(
        (example_hash(example, as_int=False) for example in dataset.data)
    )
    return _hash(hash_data, as_int=as_int)


def _hash(
    tpl: Tuple, hash_function: Callable = hashlib.sha1, as_int: bool = True
) -> Union[str, int]:
    """Deterministic hash function. The main use here is 
    providing a `commit_hash` for a Dataset to compare across
    saves/loads and ensure that operations are re-run if the hash 
    ever changes
    
    Args:
        tpl (Tuple): Tuple of data to hash
        hash_function (Callable, hashlib.sha1): Hash function from 
            python hashlib. Defaults to sha1 (same as git)
        as_int (bool, optional): Encode hash as int
    
    Returns:
        Union[str, int]: Deterministic hash using tpl data
    """
    m = hash_function()
    for e in tpl:
        if isinstance(e, str):
            e_bytes = e.encode("utf-8")
        else:
            e_bytes = bytes(e)
        m.update(e_bytes)

    hd = m.hexdigest()

    if as_int:
        return int(hd, 16)
    else:
        return hd
