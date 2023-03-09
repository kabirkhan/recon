<p align="center">
  <a href="https://kabirkhan.github.io/recon"><img src="https://raw.githubusercontent.com/kabirkhan/recon/main/docs/img/recon-ner.svg" alt="Recon"></a>
</p>
<p align="center">
    <em>Recon NER, Debug and correct annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data.</em>
</p>
<p align="center">
<a href="https://pypi.org/project/reconner" target="_blank">
    <img src="https://img.shields.io/pypi/v/reconner?style=for-the-badge" alt="PyPi Package version">
</a>
<a href="https://github.com/kabirkhan/recon/actions/workflows/ci.yml" target="_blank">
    <img alt="GitHub Actions Build badge" src="https://img.shields.io/github/actions/workflow/status/kabirkhan/recon/.github/workflows/ci.yml?branch=main&style=for-the-badge">

</a>
<a href="https://codecov.io/gh/kabirkhan/recon" rel="nofollow">
  <img alt="Codecov badge" src="https://img.shields.io/codecov/c/gh/kabirkhan/recon?style=for-the-badge" style="max-width:100%;">
</a>

<a href="https://pypi.org/project/reconner" target="_blank">
    <img src="https://img.shields.io/pypi/l/reconner?style=for-the-badge" alt="PyPi Package license">
</a>
</p>

---

**Documentation**: <a href="https://kabirkhan.github.io/recon" target="_blank">https://kabirkhan.github.io/recon</a>

**Source Code**: <a href="https://github.com/kabirkhan/recon" target="_blank">https://github.com/kabirkhan/recon</a>

---

Introducing Recon: The Ultimate Tool for Named Entity Recognition

Recon is a tool to help data scientists, machine learning engineers, and NLP researchers identify issues and gain valuable insights in their Named Entity Recognition (NER) data. With Recon, you can easily identify errors, inconsistencies, and missing data that may be impacting your NER models, and make informed decisions to improve the accuracy and efficiency of your NER pipeline.

The key features are:

* **Data Validation and Cleanup**: Easily Validate the format of your NER data. Filter overlapping Entity Annotations and fix missing properties.
* **Statistics**: Get statistics on your data. From how many annotations you have for each label, to more complicated metrics like quality scores for the balance of your dataset.
* **Model Insights**: Analyze how well your model does on your Dataset. Identify the top errors your model is making so you can prioritize data collection and correction strategically.
* **Dataset Management**: Recon provides `Dataset` and `Corpus` containers to manage the train/dev/test split of your data and apply the same functions across all splits in your data + a concatenation of all examples. Operate inplace to consistently transform your data with reliable tracking and the ability to version and rollback changes.
* **Serializable Dataset**: Serialize and Deserialize your data to and from JSON and the spaCy format. Easily extend serialization to your own custom format
* **Customization**: Recon has a simple decorator to create a function that operates on a Dataset. Easily plug in custom logic to make corrections to your data.

Whether you're a data scientist, machine learning engineer, or NLP researcher, Recon is the ultimate tool for anyone working with NER data.


## Requirements

Python 3.8 +

* <a href="https://spacy.io" class="external-link" target="_blank">spaCy</a>
* <a href="https://pydantic-docs.helpmanual.io/" class="external-link" target="_blank">Pydantic (Type system and JSON Serialization)</a>
* <a href="https://typer.tiangolo.com" class="external-link" target="_blank">Typer (CLI)</a>.


## Installation

<div class="termy">

```console
$ pip install reconner
---> 100%
Successfully installed reconner
```

</div>

## License

This project is licensed under the terms of the MIT license.
