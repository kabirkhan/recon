from typing import Any, Dict, Optional

from recon.operations import operation
from recon.types import Example, Token


@operation("recon.add_tokens.v1", pre=["recon.spacy.v1"])
def add_tokens(
    example: Example,
    *,
    use_spacy_token_ends: bool = False,
    preprocessed_outputs: Dict[str, Any],
) -> Optional[Example]:
    """Add tokens to each Example

    Args:
        example (Example): Input Example
        preprocessed_outputs (Dict[str, Any]): Outputs of preprocessors

    Returns:
        Example: Example with tokens
    """
    doc = preprocessed_outputs["recon.spacy.v1"]

    tokens = []
    token_starts = {}
    token_ends = {}

    for t in doc:
        start = t.idx
        end = t.idx + len(t)
        tokens.append(Token(text=t.text, start=start, end=end, id=t.i))
        token_starts[start] = t
        token_ends[end] = t

    example.tokens = tokens

    for span in example.spans:
        if span.start in token_starts and span.end in token_ends:
            span.token_start = token_starts[span.start].i
            if use_spacy_token_ends:
                span.token_end = token_ends[span.end].i + 1
            else:
                span.token_end = token_ends[span.end].i

        if span.token_start is None or span.token_end is None:
            return None

    return example
