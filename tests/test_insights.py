from recon.insights import (
    get_hardest_examples,
    top_label_disparities,
    top_prediction_errors,
)


def test_top_label_disparities(example_corpus, example_corpus_processed):
    top_disparities = top_label_disparities(example_corpus.all)
    assert len(top_disparities) == 8

    top_disparities = top_label_disparities(example_corpus.all, dedupe=True)
    assert len(top_disparities) == 4
    assert top_disparities[0].count == 1

    top_disparities_processed = top_label_disparities(example_corpus_processed.all)
    assert len(top_disparities_processed) == 2

    top_disparities_processed = top_label_disparities(example_corpus_processed.all, dedupe=True)
    assert len(top_disparities_processed) == 1
    assert top_disparities_processed[0].count == 2


def test_top_prediction_errors(recognizer, example_corpus):
    pred_errors = top_prediction_errors(recognizer, example_corpus.test)
    assert len(pred_errors) == 67


def test_get_hardest_examples(recognizer, example_corpus):
    hardest_examples = get_hardest_examples(recognizer, example_corpus.test)

    hardest_examples_no_count = get_hardest_examples(
        recognizer, example_corpus.test, score_count=False
    )

    assert len(hardest_examples) == len(example_corpus.test)
    assert hardest_examples[0].reference.text.startswith("Some of the free Apache Tomcat")
    assert hardest_examples_no_count[0].reference.text.startswith("Visual Basic was")
