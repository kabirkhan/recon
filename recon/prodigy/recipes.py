# isort:skip_file
# type: ignore

from collections import Counter, defaultdict
import copy
import random
from typing import Dict, Iterable, List, Optional, Union

import catalogue
import prodigy
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
import spacy
import srsly
from wasabi import msg
from recon.constants import NONE
from recon.types import HardestExample
from recon.validation import remove_overlapping_entities


def make_span_hash(span: Dict):
    return f"{span['text']}|||{span['start']}|||{span['end']}|||{span['label']}"


def get_stream_from_hardest_examples(nlp, hardest_examples: List[HardestExample]):
    for he in hardest_examples:
        task = he.example.dict()
        task = list(add_tokens(nlp, [task]))[0]
        gold_span_hashes = {make_span_hash(span): span for span in task["spans"]}
        predicted_example = None
        assert he.prediction_errors is not None
        for pe in he.prediction_errors:
            assert pe.examples is not None
            for e in pe.examples:
                if e.predicted.text == he.example.text:
                    predicted_example = e.predicted
        if predicted_example:
            pthtml = []
            predicted_example_task = predicted_example.dict()
            predicted_example_task = list(add_tokens(nlp, [predicted_example_task]))[0]

            for token in predicted_example_task["tokens"]:
                pthtml.append(f'<span class="recon-token">{token["text"]} </span>')

            pred_spans = predicted_example_task["spans"]
            pred_span_hashes = [
                make_span_hash(span) for span in predicted_example_task["spans"]
            ]

            for gold_span_hash, gold_span in gold_span_hashes.items():
                if gold_span_hash not in pred_span_hashes:
                    gold_span_miss = copy.deepcopy(gold_span)
                    gold_span_miss["label"] = NONE
                    pred_spans.append(gold_span_miss)

            pred_spans = remove_overlapping_entities(
                sorted(pred_spans, key=lambda s: s["start"])
            )
            # pred_spans = sorted(pred_spans, key=lambda s: s["start"])

            i = len(pred_spans) - 1
            while i >= 0:
                span = pred_spans[i]
                span_hash = make_span_hash(span)
                if span_hash in gold_span_hashes:
                    labelColorClass = "recon-pred-success-mark"
                elif span["label"] == NONE:
                    labelColorClass = "recon-pred-missing-mark"
                else:
                    labelColorClass = "recon-pred-error-mark"

                pthtml = (
                    pthtml[: span["token_end"] + 1]
                    + [
                        f'<span class="recon-pred-label">{span["label"]}<span class="c0178">x</span></span></span>'
                    ]
                    + pthtml[span["token_end"] + 1 :]
                )
                pthtml = (
                    pthtml[: span["token_start"]]
                    + [f'<span class="recon-pred {labelColorClass}">']
                    + pthtml[span["token_start"] :]
                )
                i -= 1
            task[
                "html"
            ] = f"""
            <h2 class='recon-title'>Recon Prediction Errors</h2>
            <h5 class='recon-subtitle'>
                The following text shows the errors your model made on this example inline.
                Correct the annotations above based on how well your model performed on this example.
                If your labeling is correct you might need to add more training examples in this domain.    
            </h5>
            <div class='recon-container'>
                {''.join(pthtml)}
            </div>
            """
        task["prediction_errors"] = [pe.dict() for pe in he.prediction_errors]
        yield task


@prodigy.recipe(
    "recon.ner_correct",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=(
        "Base model or blank:lang (e.g. blank:en) for blank model",
        "positional",
        None,
        str,
    ),
    hardest_examples=(
        "Data to annotate (file path or '-' to read from standard input)",
        "positional",
        None,
        str,
    ),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=(
        "Comma-separated label(s) to annotate or text file with one label per line",
        "option",
        "l",
        get_labels,
    ),
    exclude=(
        "Comma-separated list of dataset IDs whose annotations to exclude",
        "option",
        "e",
        split_string,
    ),
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
    log("RECIPE: Starting recipe recon.ner_correct", locals())
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

    stream = get_stream_from_hardest_examples(nlp, hardest_examples)
    stream = add_tokens(nlp, stream)  # add "tokens" key to the tasks

    table_template = """
    <style>
        .table-title {
            margin-top: -120px;
        }
        table {
            border-collapse: collapse;
        }
        table, td, th {
            border: 1px solid gray;
            vertical-align: top;
            margin-top: -50px;
        }
    </style>
    <h5 class='recon-subtitle table-title'>All prediction errors for this example.</h2>
    <table>
        <tr>
            <th>Text</th>
            <th>True Label</th>
            <th>Pred Label</th>
        </tr>
        {{#prediction_errors}}
        <tr>
            <td>{{text}}</td>
            <td>{{true_label}}</td>
            <td>{{pred_label}}</td>
        </tr>
        {{/prediction_errors}}
    </table>
    """

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "config": {
            "lang": nlp.lang,
            "labels": labels.split(","),
            "exclude_by": "input",
            "blocks": [
                {"view_id": "ner_manual"},
                {"view_id": "html"},
                {"view_id": "html", "field_rows": 3, "html_template": table_template},
            ],
            "global_css": """
            .recon-title {
                text-align: left;
                margin-top: -80px;
            }
            .recon-subtitle {
                text-align: left;
                margin-top: -80px;
                white-space: normal;
            }
            .recon-container {
                text-align: left;
                line-height: 2;
                margin-top: -80px;
                white-space: pre-line;
            }
            .recon-pred {
                color: inherit;
                margin: 0 0.15em;
                display: inline;
                padding: 0.25em 0.4em;
                font-weight: bold;
                line-height: 1;
                -webkit-box-decoration-break: clone;
            }
            .recon-pred-success-mark {
                background: #00cc66;
            }
            .recon-pred-error-mark {
                background: #fc7683;
            }
            .recon-pred-missing-mark {
                background: #84b4c4;
            }
            .recon-pred-label { 
                color: #583fcf;
                font-size: 0.675em;
                font-weight: bold;
                font-family: "Roboto Condensed", "Arial Narrow", sans-serif;
                margin-left: 8px;
                text-transform: uppercase;
                vertical-align: middle;
            }
            """,
        },
    }
