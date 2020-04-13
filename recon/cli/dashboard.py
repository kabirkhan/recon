# isort:skip_file
# type: ignore
# -*- coding: utf-8 -*-
import typer
from pathlib import Path
from typing import Callable

from recon.corpus import Corpus
from recon.stats import (
    calculate_label_balance_entropy,
    calculate_label_distribution_similarity,
    get_ner_stats,
)
from recon.types import Example, NERStats
from wasabi import Printer
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


def dashboard(data_dir: Path) -> None:
    """Calculate statistics on a Corpus

    Args:
        data_dir (Path): Path to data folder
    """
    msg: Printer = Printer()

    # with msg.loading("Loading Corpus from Disk"):
    corpus = Corpus.from_disk(data_dir)
    # msg.good("Done")

    ner_stats = corpus.apply(get_ner_stats)

    # external_stylesheets = [
    #     "https://codepen.io/chriddyp/pen/bWLwgP.css",
    #     "https://cdn.jsdelivr.net/npm/uikit@3.3.7/dist/css/uikit.min.css"
    # ]

    # external_scripts = [
    #     "https://cdn.jsdelivr.net/npm/uikit@3.3.7/dist/js/uikit.min.js",
    #     "https://cdn.jsdelivr.net/npm/uikit@3.3.7/dist/js/uikit-icons.min.js"
    # ]

    # app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

    # def generate_bar_chart_stats(id: str, ner_stats: NERStats, name: str = None):
    #     return dcc.Graph(
    #         id=id,
    #         figure={
    #             'data': [
    #                 go.Bar(
    #                     x = list(ner_stats.n_annotations_per_type.values()),
    #                     y = list(ner_stats.n_annotations_per_type.keys()),
    #                     orientation='h'
    #                 )
    #             ],
    #             'layout': {
    #                 'title': name or id.capitalize()
    #             }
    #         }
    #     )

    # app.layout = html.Div(children=[
    #     html.Div(className="uk-child-width-1-2@s uk-grid-match")
    #     html.H1(className="" children='Recon NER Dashboard'),

    #     html.Div(children='''
    #         This dashboard shows statistics for all your data.
    #     '''),

    #     html.Div(children=[
    #         generate_bar_chart_stats("train", ner_stats["train"]),
    #         generate_bar_chart_stats("dev", ner_stats["dev"]),
    #         generate_bar_chart_stats("test", ner_stats["test"]),
    #         generate_bar_chart_stats("all", ner_stats["all"])
    #     ], style={'columnCount': 4})
    # ])

    uvicorn.run(app, port=9090)

    # app.run_server(debug=True)


if __name__ == "__main__":
    typer.run(dashboard)
