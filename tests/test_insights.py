from recon.insights import (
    get_hardest_examples,
    top_label_disparities,
    top_prediction_errors,
)


def test_top_label_disparities(example_corpus):
    top_disparities = top_label_disparities(example_corpus.all)
    assert len(top_disparities) == 2

    top_disparities = top_label_disparities(example_corpus.all, dedupe=True)
    assert len(top_disparities) == 1
    assert top_disparities[0].count == 2


def test_top_prediction_errors(recognizer, example_corpus):
    pred_errors = top_prediction_errors(recognizer, example_corpus.test)
    assert len(pred_errors) == 67


def test_get_hardest_examples(recognizer, example_corpus):
    pred_errors = top_prediction_errors(recognizer, example_corpus.test)
    hardest_examples = get_hardest_examples(pred_errors)

    assert len(hardest_examples) == 42
    assert hardest_examples[0].example.text.startswith(
        "Some of the free Apache Tomcat resources"
    )
