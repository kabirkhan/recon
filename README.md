<p align="center">
    <em>ReconNER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data.</em>
</p>
<p align="center">
<a href="https://dev.azure.com/kabirkhan1137/reconner/_build" target="_blank">
    <img src="https://dev.azure.com/kabirkhan1137/reconner/_apis/build/status/kabirkhan.reconner?branchName=master" alt="Build Status">
</a>
<a href="https://codecov.io/gh/microsoft/reconner">
  <img src="https://codecov.io/gh/microsoft/reconner/branch/master/graph/badge.svg" alt="Coverage"/>
</a>
<a href="https://pypi.org/project/reconner" target="_blank">
    <img src="https://badge.fury.io/py/reconner.svg" alt="Package version">
</a>
</p>

---

**Documentation**: <a href="https://kabirkhan.github.io/reconner" target="_blank">https://kabirkhan.github.io/reconner</a>

**Source Code**: <a href="https://github.com/kabirkhan/reconner" target="_blank">https://github.com/kabirkhan/reconner</a>

---

ReconNER is a library to help you fix your annotated NER data and identify examples that are hardest for your model to predict so you can strategically prioritize the examples you annotate. 

The key features are:

* **Data Validation and Cleanup**: Easily Validate the format of your NER data. Filter overlapping Entity Annotations, fix missing properties.
* **Model Insights**: Analyze how well your model does on your Dataset. Identify the top errors your model is making so you can prioritize data collection and correction strategically.
* **Model Insights**: Analyze how well your model does on your Dataset. Identify the top errors your model is making so you can prioritize data collection and correction strategically.
* **Dataset Management**: `reconner` provides a `Dataset` class to manage the train/dev/test split of your data and apply the same functions across all splits in your data + a concatenation of all examples. Operate inplace to consistently transform your data.
* **Serializable Dataset**: Serialize and Deserialize your data to and from JSON to the `reconner` type system. 
* **Type Hints**: Comprehensive Typing system based on Python 3.6+ Type Hints

## Requirements

Python 3.6+

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
