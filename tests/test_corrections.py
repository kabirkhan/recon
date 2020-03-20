from recon.corpus import Corpus
from recon.corrections import fix_annotations, rename_labels
from recon.insights import get_label_disparities
from recon.types import Example


def test_rename_labels(example_corpus):
    example_corpus.apply_(rename_labels, {"SKILL": "skill"})
    for e in example_corpus.test:
        for s in e.spans:
            if s.label == "SKILL":
                raise ValueError("Did not rename label properly", e)

    assert isinstance(example_corpus.train[0], Example)
    assert isinstance(example_corpus.dev[0], Example)
    assert isinstance(example_corpus.test[0], Example)


def test_fix_annotations(example_corpus):
    disparities = get_label_disparities(
        example_corpus.all, label1="SKILL", label2="JOB_ROLE"
    )
    assert disparities == {"model", "software development engineer"}

    example_corpus.apply_(
        fix_annotations, {"software development engineer": "JOB_ROLE"}
    )

    disparities_fixed = get_label_disparities(
        example_corpus.all, label1="SKILL", label2="JOB_ROLE"
    )
    assert disparities_fixed == {"model"}
