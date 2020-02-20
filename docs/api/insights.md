# `reconner.insights`

The `reconner.insights` module provides more complex functionality for understanding your dataset.
It provides functions for identifying disparities in your annotations and identifying the kinds of examples and labels
that are hardest for your model to identify.

Some of the functionality in `reconner.insights` require a `reconner.recognizer.EntityRecognizer` object.
You can read more about the `EntityRecognizer` class here: [Tutorial - Custom EntityRecognizer](../tutorial/custom_entity_recognizer.md)

---
## API

::: reconner.insights.ents_by_label
    :docstring:

::: reconner.insights.get_label_disparities
    :docstring:

::: reconner.insights.top_prediction_errors
    :docstring:
