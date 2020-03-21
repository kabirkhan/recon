# isort:skip_file

import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import spacy
import streamlit as st
import typer
from wasabi import Printer
from recon import Corpus
from recon.constants import NONE
from recon.corrections import rename_labels
from recon.insights import (
    get_ents_by_label,
    get_hardest_examples,
    get_label_disparities,
    top_label_disparities,
    top_prediction_errors,
)
from recon.recognizer import SpacyEntityRecognizer
from recon.stats import get_entity_coverage, get_ner_stats
from recon.types import Example, HardestExample, PredictionError


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)

@st.cache
def load_dataset(data_dir: Path):
    return Corpus.from_disk(data_dir)


st.sidebar.title("Interactive Recon Visualizer")
st.sidebar.markdown(
    """
Process your NER data with [Recon](https://microsoft.github.io/reconner)
"""
)

spacy_model = st.sidebar.text_input("Model name or path")
dataset_dir = st.sidebar.text_input("Data Directory")

nlp = load_model(spacy_model)
ds = load_dataset(dataset_dir)

st.header("Stats")
for name, stats in ds.apply(get_ner_stats).items():
    if 'examples_with_type' in stats:
        del stats['examples_with_type']
    st.subheader(f"{name} data - {stats['n_examples']} Total Examples")
    ents_per_type = stats['ents_per_type']

    chart_data = pd.DataFrame(
        np.asarray(list(zip(ents_per_type.keys(), ents_per_type.values()))),
        columns=["Label", "N Annotations"]
    ).sort_values("Label")
    fig = px.bar(chart_data, x="N Annotations", y="Label", orientation="h")
    st.plotly_chart(fig)
    st.json(stats)



# def main(model: str = typer.Option(...), data_dir: Path = typer.Option(...)):
#     nlp = load_model(model)
#     ds = load_dataset(data_dir)

#     st.text(get_ner_stats(ds.test, serialize=True, no_print=True))


# if __name__ == "__main__":
#     typer.run(main)
