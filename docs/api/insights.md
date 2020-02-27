# `recon.insights`

The `recon.insights` module provides more complex functionality for understanding your dataset.
It provides functions for identifying disparities in your annotations and identifying the kinds of examples and labels
that are hardest for your model to identify.

Some of the functionality in `recon.insights` require a `recon.recognizer.EntityRecognizer` object.
You can read more about the `EntityRecognizer` class here: [Tutorial - Custom EntityRecognizer](../tutorial/custom_entity_recognizer.md)

---
## API

::: recon.insights.ents_by_label
    :docstring:

::: recon.insights.get_label_disparities
    :docstring:

::: recon.insights.top_prediction_errors
    :docstring:
