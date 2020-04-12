# isort:skip_file
# type: ignore

from collections import Counter, defaultdict
import copy
import random
from typing import Any, Dict, Iterable, List, Optional, Union

import catalogue
import prodigy
from prodigy.components.db import connect
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import add_tokens
from prodigy.models.matcher import PatternMatcher
from prodigy.recipes.ner import get_labels_from_ner
from prodigy.util import (
    INPUT_HASH_ATTR,
    TASK_HASH_ATTR,
    get_labels,
    log,
    set_hashes,
    split_string,
)
import spacy
import srsly
from wasabi import msg
from recon.constants import NONE
from recon.loaders import read_jsonl
from recon.types import HardestExample, Example, Span
from recon.validation import remove_overlapping_entities

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(
            children="""
        Dash: A web application framework for Python.
    """
        ),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "Montréal"},
                ],
                "layout": {"title": "Dash Data Visualization"},
            },
        ),
    ]
)


@prodigy.recipe(
    "recon.ner_manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy model for tokenization or blank:lang (e.g. blank:en)", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    api=("DEPRECATED: API loader to use", "option", "a", str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    patterns=("Path to match patterns file", "option", "pt", str),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def manual(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]] = "-",
    api: Optional[str] = None,
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    patterns: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Mark spans by token. Requires only a tokenizer and no entity recognizer,
    and doesn't do any active learning.
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
    stream = get_stream(source, api, loader, rehash=True, dedup=True, input_key="text")
    if patterns is not None:
        pattern_matcher = PatternMatcher(nlp, combine_matches=True, all_examples=True)
        pattern_matcher = pattern_matcher.from_disk(patterns)
        stream = (eg for _, eg in pattern_matcher(stream))
    stream = add_tokens(nlp, stream)  # add "tokens" key to the tasks

    print(app.index())

    print(app.server.url_map)
    import requests

    # print(app.server.view_functions['/'])
    print(app.serve_layout())
    # app.run_server()
    # html = requests.get('127.0.0.1:8050/').text()

    # with open('./recon/prodigy/templates/graph.html') as f:
    #     html = f.read()
    html = ""

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "blocks": [{"view_id": "html", "html_template": html}, {"view_id": "ner_manual"}],
        },
    }
