import copy
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Union

import catalogue
import prodigy
import spacy
import srsly
from prodigy.components.db import connect
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import add_tokens
from prodigy.recipes.ner import get_labels_from_ner
from prodigy.util import (
    INPUT_HASH_ATTR,
    TASK_HASH_ATTR,
    get_labels,
    log,
    set_hashes,
    split_string,
)
from recon.types import HardestExample
from wasabi import msg


def get_stream_from_hardest_examples(hardest_examples: List[HardestExample]):
    for he in hardest_examples:
        task = he.example.dict()
        task['prediction_errors'] = [pe.dict() for pe in he.prediction_errors]
        yield task


@prodigy.recipe(
    "recon.ner_correct",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Base model or blank:lang (e.g. blank:en) for blank model", "positional", None, str),
    hardest_examples=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
)
def ner_correct(
    dataset: str,
    spacy_model: str,
    hardest_examples: List[HardestExample],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
):
    """
    Stream a List of `recon.types.HardestExample` instances to prodigy
    for review/correction. Uses the Prodigy blocks interface to display
    prediction error information along with ner view
    """
    log("RECIPE: Starting recipe ner.manual", locals())
    if spacy_model.startswith("blank:"):
        nlp = spacy.blank(spacy_model.replace("blank:", ""))
    else:
        nlp = spacy.load(spacy_model)
    labels = label  # comma-separated list or path to text file
    if not labels:
        labels = get_labels_from_ner(nlp)
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    # stream = get_stream(source, None, loader, rehash=True, dedup=True, input_key="text")

    stream = get_stream_from_hardest_examples(hardest_examples)
    stream = add_tokens(nlp, stream)  # add "tokens" key to the tasks

    html_template = """
    <ul>
    {% for pe in prediction_errors %}
        <li>{{pe.text}} {{pe.true_label}}</li>
    {% endfor %}
    </ul>
    """

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "blocks": [
                {"view_id": "ner_manual"},
                {"view_id": "html", "field_rows": 3, "html_template": html_template},
            ]
        }
    }
