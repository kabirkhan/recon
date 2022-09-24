# isort:skip_file
# type: ignore

from typing import Any, Dict, Iterable, List, Optional, Union

import prodigy
from prodigy.components.db import connect
from prodigy.components.loaders import get_stream
from prodigy.components.preprocess import add_tokens
from prodigy.util import (
    get_labels,
    log,
    split_string,
)
import spacy
from wasabi import msg
from recon.dataset import Dataset
from recon.insights import get_hardest_examples
from recon.operations.core import operation
from recon.recognizer import SpacyEntityRecognizer
from recon.types import Example, HardestExample, Span


@prodigy.recipe(
    "recon.ner-correct.v1",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy model for tokenization or blank:lang (e.g. blank:en)", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def recon_ner_correct_v1(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:

    log("RECIPE: Starting recipe ner.manual", locals())
    nlp = spacy.load(spacy_model)
    labels = label  # comma-separated list or path to text file
    if not labels:
        labels = nlp.pipe_labels.get("ner", [])
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(source, loader=loader, rehash=True, dedup=True, input_key="text")
    # Add "tokens" key to the tasks, either with words or characters
    stream = add_tokens(nlp, stream)
    stream = [Example(**eg) for eg in stream]

    rec = SpacyEntityRecognizer(nlp)
    hes = get_hardest_examples(rec, stream)

    stream = _stream_from_hardest_examples(hes)

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
            ],
            "javascript": """

            const disablePredInteraction = event => {
                let xpath = "//span[contains(text(),':PREDICTED')]";
                let matchingElement = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                let length = matchingElement.snapshotLength;

                for (var i = 0; i < length; i++) {
                    matchingElement.snapshotItem(i).parentElement.parentElement.onclick = function(e) { e.stopImmediatePropagation(); }
                }
            }

            document.addEventListener('prodigyanswer', disablePredInteraction)
            """,
        },
    }


@prodigy.recipe(
    "recon.ner-merge.v1",
    # fmt: off
    dataset=("Dataset with saved annotations to from recon.ner_correct", "positional", None, str),
    recon_dataset=("Recon dataset name", "positional", None, str),
    source=("Source data to merge examples with (file path or List of examples)", "positional", None, str),
    output_dir=("Optional output directory to save dataset to", "positional", None, str),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    # fmt: on
)
def recon_ner_merge_v1(
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
    dataset.apply_("recon.prodigy.merge_examples", prodigy_texts_to_examples)
    assert len(dataset) == prev_len

    if output_dir:
        log(f"RECIPE: Fixing {len(prodigy_examples)} examples in data")
        dataset.to_disk(output_dir)


@operation("recon.prodigy.merge_examples.v1")
def merge_examples(example, prodigy_texts_to_examples):
    if example.text in prodigy_texts_to_examples:
        return prodigy_texts_to_examples[example.text]
    else:
        return example


def _stream_from_hardest_examples(hes: List[HardestExample]):
    for he in hes:
        combined = he.reference.copy(deep=True)
        annot_spans = [
            Span(**span.dict(exclude={"source"}), source="ref") for span in he.reference.spans
        ]
        pred_spans = [
            Span(
                **span.dict(exclude={"source", "label"}), source="pred", label=f"{span.label}:PRED"
            )
            for span in he.prediction.spans
        ]
        combined.spans = sorted(annot_spans + pred_spans, key=lambda s: s.start)
        task = combined.dict()
        yield task
