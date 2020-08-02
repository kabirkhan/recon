<p align="center">
  <a href="https://kabirkhan.github.io/recon"><img src="https://kabirkhan.github.io/recon/img/recon.svg" alt="Recon"></a>
</p>
<p align="center">
    <em>Recon NER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data.</em>
</p>
<p align="center">
<a href="https://pypi.org/project/reconner" target="_blank">
    <img src="https://img.shields.io/pypi/v/reconner?style=for-the-badge" alt="PyPi Package version">
</a>
<a href="https://dev.azure.com/kabirkhan1137/ReconNER/_build?definitionId=4" target="_blank">
    <img alt="Azure Pipelines Build badge" src="https://img.shields.io/azure-devops/build/kabirkhan1137/ReconNER/kabirkhan.recon?style=for-the-badge">
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

ReconNER is a library to help you fix your annotated NER data and identify examples that are hardest for your model to predict so you can strategically prioritize the examples you annotate.

The key features are:

* **Data Validation and Cleanup**: Easily Validate the format of your NER data. Filter overlapping Entity Annotations, fix missing properties.
* **Model Insights**: Analyze how well your model does on your Dataset. Identify the top errors your model is making so you can prioritize data collection and correction strategically.
* **Model Insights**: Analyze how well your model does on your Dataset. Identify the top errors your model is making so you can prioritize data collection and correction strategically.
* **Dataset Management**: ReconNER provides a `Dataset` class to manage the train/dev/test split of your data and apply the same functions across all splits in your data + a concatenation of all examples. Operate inplace to consistently transform your data.
* **Serializable Dataset**: Serialize and Deserialize your data to and from JSON to the ReconNER type system.
* **Type Hints**: Comprehensive Typing system based on Python 3.6+ Type Hints

## Requirements

Python 3.6+

`ReconNER` is built on a few comprehensive, high-performing packages.

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
