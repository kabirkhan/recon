# isort:skip_file
# type: ignore

from typing import Dict, List, Optional, Union

import prodigy
from prodigy.components.db import connect
from prodigy.util import (
    get_labels,
    log,
    split_string,
)
import spacy
from wasabi import msg
from recon.dataset import Dataset
from recon.operations.core import operation
from recon.types import HardestExample, Example, Span


def make_span_hash(span: Union[Span, Dict[str, object]]) -> str:
    if isinstance(span, Span):
        h = f"{span.text}|||{span.start}|||{span.end}|||{span.label}"
    else:
        h = f'{span["text"]}|||{span["start"]}|||{span["end"]}|||{span["label"]}'
    return h


def get_stream_from_hardest_examples(nlp, hardest_examples: List[HardestExample]):
    for he in hardest_examples:
        task = he.example.dict()
        gold_span_hashes = {make_span_hash(span): span for span in he.example.spans}
        predicted_example = None
        assert he.prediction_errors is not None
        for pe in he.prediction_errors:
            assert pe.examples is not None
            for e in pe.examples:
                if e.predicted.text == he.example.text:
                    predicted_example = e.predicted
            if predicted_example:
                pred_spans = []
                for span in predicted_example.spans:
                    if span.label != "NOT_LABELED":
                        span_ = span.copy(deep=True)
                        span_.label = f"{span.label}:PREDICTED"
                        pred_spans.append(span_.dict())

                print("PREDICTED SPANS", pred_spans)
                task["spans"] = sorted(task["spans"] + pred_spans, key=lambda s: s["start"])
                task[
                    "html"
                ] = f"""
                <h2 class='recon-title'>Recon Prediction Errors</h2>
                <h5 class='recon-subtitle'>
                    The following text shows the errors your model made on this example inline.
                    Correct the annotations above based on how well your model performed on this example.
                    If your labeling is correct you might need to add more training examples in this domain.    
                </h5>
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
        labels = nlp.get_pipe("ner").labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")

    log(f"RECIPE: Annotating with {len(labels)} labels", labels)

    stream = get_stream_from_hardest_examples(nlp, hardest_examples)

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

    def before_db(examples):
        for eg in examples:
            new_spans = []
            for span in eg["spans"]:
                if not span["label"].endswith(":PREDICTED"):
                    new_spans.append(span)
            eg["spans"] = new_spans
        return examples

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "before_db": before_db,
        "config": {
            "lang": nlp.lang,
            "labels": labels,
            "exclude_by": "input",
            "force_stream_order": True,
            "overlapping_spans": True,
            "blocks": [
                {"view_id": "spans_manual", "overlapping_spans": True},
                {"view_id": "html"},
                {"view_id": "html", "field_rows": 3, "html_template": table_template},
            ],
            "global_css": f"""
            .prodigy-container {{
                max-width: {'900px' if len(labels) > 8 else '600px'}
            }}
            .recon-title {{
                text-align: left;
                margin-top: -80px;
            }}
            .recon-subtitle {{
                text-align: left;
                margin-top: -80px;
                white-space: normal;
            }}
            .recon-container {{
                text-align: left;
                line-height: 2;
                margin-top: -80px;
                white-space: pre-line;
            }}
            .recon-pred {{
                color: inherit;
                margin: 0 0.15em;
                display: inline;
                padding: 0.25em 0.4em;
                font-weight: bold;
                line-height: 1;
                -webkit-box-decoration-break: clone;
            }}
            .recon-pred-success-mark {{
                background: #00cc66;
            }}
            .recon-pred-error-mark {{
                background: #fc7683;
            }}
            .recon-pred-missing-mark {{
                background: #84b4c4;
            }}
            .recon-pred-label {{
                color: #583fcf;
                font-size: 0.675em;
                font-weight: bold;
                font-family: "Roboto Condensed", "Arial Narrow", sans-serif;
                margin-left: 8px;
                text-transform: uppercase;
                vertical-align: middle;
            }}
            """,
        },
    }


@prodigy.recipe(
    "recon.ner_merge",
    # fmt: off
    dataset=("Dataset with saved annotations to from recon.ner_correct", "positional", None, str),
    recon_dataset=("Recon dataset name", "positional", None, str),
    source=("Source data to merge examples with (file path or List of examples)", "positional", None, str),
    output_dir=("Optional output directory to save dataset to", "positional", None, str),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def ner_merge(
    dataset: str,
    recon_dataset: str,
    source: Union[str, Dataset],
    output_dir: Optional[str] = None,
    exclude: Optional[List[str]] = None,
):
    """
    Stream a List of `recon.types.HardestExample` instances to prodigy
    for review/correction. Uses the Prodigy blocks interface to display
    prediction error information along with ner view
    """
    log("RECIPE: Starting recipe recon.ner_merge", locals())
    if isinstance(source, str):
        dataset = Dataset(recon_dataset).from_disk(source)
    else:
        dataset = source

    DB = connect()
    if dataset not in DB:
        msg.fail(f"Can't find dataset '{dataset}'", exits=1)

    prodigy_raw_examples = DB.get_dataset(dataset)
    prodigy_examples = [Example(**eg) for eg in prodigy_raw_examples if eg["answer"] == "accept"]
    prodigy_texts_to_examples = {e.text: e for e in prodigy_examples}

    prev_len = len(dataset)
    dataset.apply_("recon.v1.prodigy.merge_examples", prodigy_texts_to_examples)
    assert len(dataset) == prev_len

    if output_dir:
        log(f"RECIPE: Fixing {len(prodigy_examples)} examples in data")
        dataset.to_disk(output_dir)


@operation("recon.v1.prodigy.merge_examples")
def merge_examples(example, prodigy_texts_to_examples):
    if example.text in prodigy_texts_to_examples:
        return prodigy_texts_to_examples[example.text]
    else:
        return example
