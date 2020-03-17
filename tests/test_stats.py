from recon.stats import get_ner_stats


def test_get_ner_stats(example_data):
    stats = get_ner_stats(example_data["train"])
    str_stats = get_ner_stats(example_data["train"], serialize=True)
    assert isinstance(str_stats, str)

    assert stats.n_examples == 102
    assert stats.n_examples_no_entities == 29
    assert stats.n_annotations_per_type == {"SKILL": 191, "PRODUCT": 34, "JOB_ROLE": 5}
