from typing import TYPE_CHECKING, Callable, Tuple

import xxhash

if TYPE_CHECKING:
    from recon.dataset import Dataset
    from recon.types import Example, PredictionError, Span, Token


def token_hash(token: "Token") -> int:
    """Hash of Token type

    Args:
        token (Token): Token to hash

    Returns:
        str: Token hash
    """
    return _hash((token.text, token.start, token.end, token.id))


def span_hash(span: "Span") -> int:
    """Hash of Span type

    Args:
        span (Span): Span to hash

    Returns:
        str: Span hash
    """
    hash_data = (
        span.start,
        span.end,
        span.label,
        span.text,
        span.token_start if span.token_start else 0,
        span.token_end if span.token_end else 0,
    )
    return _hash(hash_data)


def example_hash(example: "Example") -> int:
    """Hash of Example type

    Args:
        example (Example): Example to hash

    Returns:
        str: Example hash
    """
    hash_data = [example.text]
    for span in example.spans:
        hash_data += [
            span.start,
            span.end,
            span.label,
            span.text,
        ]
    return _hash(tuple(hash_data))


def tokenized_example_hash(example: "Example") -> int:
    """Hash of Example type including token data

    Args:
        example (Example): Example to hash

    Returns:
        str: Example hash
    """
    tokens = example.tokens or []
    hash_data = [example.text]
    for span in example.spans:
        hash_data += [
            span.start,
            span.end,
            span.label,
            span.text,
            span.token_start if span.token_start else 0,
            span.token_end if span.token_end else 0,
        ]
    for token in tokens:
        hash_data += [token.text, token.start, token.end, token.id]

    return _hash(tuple(hash_data))


def dataset_hash(dataset: "Dataset") -> int:
    """Hash of Dataset

    Args:
        dataset (Dataset): Dataset to hash

    Returns:
        str: Dataset hash
    """
    hash_data = (dataset.name,) + tuple(
        (example_hash(example) for example in dataset.data)
    )
    return _hash(hash_data)


def prediction_error_hash(prediction_error: "PredictionError") -> int:
    """Hash of PredictionError

    Args:
        prediction_error (PredictionError): PredictionError to hash

    Returns:
        str: PredictionError hash
    """
    hash_data = (
        prediction_error.text,
        prediction_error.true_label,
        prediction_error.pred_label,
    )
    return _hash(hash_data)


def _hash(tpl: Tuple, hash_function: Callable = xxhash.xxh3_64) -> int:
    """Deterministic hash function. The main use here is
    providing a `commit_hash` for a Dataset to compare across
    saves/loads and ensure that operations are re-run if the hash
    ever changes

    Args:
        tpl (Tuple): Tuple of data to hash
        hash_function (Callable, hashlib.sha1): Hash function from
            python hashlib. Defaults to sha1 (same as git)

    Returns:
        str: Deterministic hash using tpl data
    """
    m = hash_function()
    for item in tpl:
        if isinstance(item, str):
            item_data = item.encode("utf-8")
        elif isinstance(item, int):
            item_data = str(item).encode("utf-8")
        else:
            item_data = bytes(item)
        m.update(item_data)
    return m.intdigest()
