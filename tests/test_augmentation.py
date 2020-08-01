from recon.dataset import Dataset
from recon.types import Example, Span


def test_ent_label_substitution():
    example = Example(
        text="This is a first sentence with entity. This is an entity in the 2nd sentence.",
        spans=[
            Span(text="entity", start=30, end=36, label="ENTITY"),
            Span(text="entity", start=49, end=55, label="ENTITY"),
        ],
    )

    ds = Dataset("test_dataset", data=[example])
    ds.apply_("recon.v1.augment.ent_label_sub", label="ENTITY", subs=["new entity"], sub_prob=1.0)

    assert len(ds) == 2

    assert ds.data[0] == example
    assert ds.data[1] == Example(
        text="This is a first sentence with new entity. This is an new entity in the 2nd sentence.",
        spans=[
            Span(text="new entity", start=30, end=40, label="ENTITY"),
            Span(text="new entity", start=53, end=63, label="ENTITY"),
        ],
    )
