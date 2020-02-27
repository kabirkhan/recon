from recon.stats import ner_stats


def test_ner_stats(example_data):
    stats = ner_stats(example_data["train"])
    printed_stats = ner_stats(example_data["train"], serialize=True)
    str_stats = ner_stats(example_data["train"], serialize=True, no_print=True)
    assert printed_stats is None
    assert isinstance(str_stats, str)

    assert stats["n_examples"] == 102
    assert stats["n_examples_no_entities"] == 29
    assert stats["ents_per_type"] == {"SKILL": 191, "PRODUCT": 34, "JOB_ROLE": 5}
