import copy
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from spacy.language import Language

from .dataset import Dataset
from .operations import op_iter, operation
from .registry import tokenizers
from .types import (
    Example,
    OperationState,
    Span,
    Token,
    Transformation,
    TransformationCallbacks,
    TransformationType,
)


@operation("fix_tokenization_and_spacing", batch=True)
def fix_tokenization_and_spacing(
    examples: List[Example],
    *,
    callbacks: TransformationCallbacks = TransformationCallbacks(),
    tokenizer: str = "default",
    verbose: bool = False,
) -> List[Example]:
    """Fix tokenization and spacing issues where there are annotation spans that 
    don't fall on a token boundary. This can happen if annotations are done at the
    character level, not the token level. Often, when scraping web text it's easy to
    get two words pushed together where one is an entity so this can fix a lot of issues.
    
    Args:
        examples (List[Example]): List of examples
        tokenizer (str, optional): Name of tokenizer in tokenizers registry to use
        verbose (bool, optional): Print status
    
    Returns:
        List[Example]: List of examples with fixed tokenization
    """
    fixed_examples = []
    tokenization_errors: List[Tuple[Example, Span]] = []
    unfixable_examples: Set[str] = set()

    nlp = tokenizers.get("default")()
    texts = (e.text for e in examples)

    with nlp.disable_pipes(*nlp.pipe_names):
        for (orig_example_hash, example), doc in zip(op_iter(examples), nlp.pipe(texts)):
            doc = nlp.make_doc(example.text)

            tokens = []
            token_starts = {}
            token_ends = {}

            for t in doc:
                start = t.idx
                end = t.idx + len(t)
                tokens.append(Token(text=t.text, start=start, end=end, id=t.i))
                token_starts[start] = t
                token_ends[end] = t

            spans_to_increment: Dict[int, int] = defaultdict(int)
            for span_i, span in enumerate(example.spans):
                if span.start in token_starts and span.end in token_ends:
                    # Aligns to token boundaries, nothing to change here
                    continue

                if span.start in token_starts and span.end not in token_ends:
                    # Span start aligns to token_start but end doesn't
                    # e.g. [customer][PERSONTYPE]s but should be annotated as [customers][PERSONTYPE]
                    tokenization_errors.append((example, span))
                    # print("BAD END")
                    if span.end + 1 in token_ends:
                        # Likely off by 1 annotation
                        # e.g. [customer][PERSONTYPE]s but should be annotated as [customers][PERSONTYPE]
                        span.end += 1
                        span.text = example.text[span.start : span.end]
                        # print("SPAN CORRECTED OFF BY 1", example.text, span)
                    elif span.end - 1 in token_ends:
                        span.end -= 1
                        span.text = example.text[span.start : span.end]
                    else:
                        # Likely bad tokenization
                        # e.g. [Quadling][GPE]Country should be split to [Quadling][GPE] Country
                        for j in range(span_i + 1, len(example.spans)):
                            spans_to_increment[j] += 1
                        fe_text = example.text

                        split_start = span.start
                        if (
                            len(spans_to_increment) > 1
                            and span_i != list(spans_to_increment.keys())[0]
                        ):
                            split_start += spans_to_increment.get(span_i, 0)
                        split_end = span.end
                        if (
                            len(spans_to_increment) > 1
                            and span_i != list(spans_to_increment.keys())[0]
                        ):
                            split_end += spans_to_increment.get(span_i, 0)
                        new_text = f"{fe_text[:split_start]}{span.text} {fe_text[split_end:]}"

                        example.text = new_text

                elif span.start not in token_starts and span.end in token_ends:
                    # Bad tokenization
                    # e.g. with[Raymond][PERSON] but text should be split to with [Raymond][PERSON]
                    # print("BAD START", span.text)
                    tokenization_errors.append((example, span))
                    for j in range(span_i, len(example.spans)):
                        spans_to_increment[j] += 1

                    fe_text = example.text

                    split_start = span.start
                    if len(spans_to_increment) > 1 and span_i != list(spans_to_increment.keys())[0]:
                        split_start += spans_to_increment.get(span_i, 0)
                    split_end = span.end
                    if len(spans_to_increment) > 1 and span_i != list(spans_to_increment.keys())[0]:
                        split_end += spans_to_increment.get(span_i, 0)

                    new_text = f"{fe_text[:split_start]} {span.text}{fe_text[split_end:]}"
                    example.text = new_text
                else:
                    # Something is super fucked up.
                    # print("SPAN CORRECTED OFF BY 1 unfixable", example.text, span)
                    before = span.start
                    after = span.end
                    tokenization_errors.append((example, span))

                    # if (before >= 0 and after < len(span.text) and span[before] not in token_starts and span[before] != ' ' and span[after] not in token_ends and span[after] != ' '):
                    #     fe_text = example.text
                    #     new_text = f"{fe_text[:span.start]} {span.text}{fe_text[span.end:]}"
                    #     spans_to_increment[span_i] += 1
                    #     for j in range(span_i + 1, len(example.spans)):
                    #         spans_to_increment[j] += 2
                    # else:
                    unfixable_examples.add(example.text)
                    break

                # Increment the start and end characters for each span
            for span_i, count in spans_to_increment.items():
                example.spans[span_i].start += count
                example.spans[span_i].end += count

            if example.text not in unfixable_examples:
                callbacks.track_example(orig_example_hash, example)
                fixed_examples.append(example)
            else:
                callbacks.remove_example(orig_example_hash)

    if tokenization_errors and verbose:
        print(f"Found {len(tokenization_errors)} tokenization errors.")
        print(f"Found {len(unfixable_examples)} unfixable tokenization errors.")

    return fixed_examples


@operation("add_tokens", batch=True)
def add_tokens(
    examples: List[Example],
    *,
    callbacks: TransformationCallbacks = TransformationCallbacks(),
    tokenizer: str = "default",
    verbose: bool = False,
) -> List[Example]:
    """Add tokens to each Example
    
    Args:
        examples (List[Example]): List of examples
        tokenizer (str, optional): Name of tokenizer in tokenizers registry to use
        verbose (bool, optional): Print status
    
    Returns:
        List[Example]: List of examples with tokens
    """
    output_examples: List[Example] = []
    tokenization_errors: List[Tuple[Example, Span]] = []
    unfixable_examples: Set[str] = set()
    nlp = tokenizers.get(tokenizer)()
    texts = (e.text for e in examples)

    with nlp.disable_pipes(*nlp.pipe_names):
        for (orig_example_hash, example), doc in zip(op_iter(examples), nlp.pipe(texts)):
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
                    span.token_end = token_ends[span.end].i

                if span.token_start is None or span.token_end is None:
                    tokenization_errors.append((example, span))
                    unfixable_examples.add(example.text)

            if example.text not in unfixable_examples:
                callbacks.track_example(orig_example_hash, example)
                output_examples.append(example)
            else:
                callbacks.remove_example(orig_example_hash)

    if verbose:
        print(f"Found {len(tokenization_errors)} tokenization errors.")
        print(f"Found {len(unfixable_examples)} unfixable examples.")

    return output_examples
