from recon.augmentation import ent_label_sub  # noqa
from recon.dataset import Dataset
from recon.types import Example, Span


def test_ent_label_substitution():
    example = Example(
        text=(
            "This is a first sentence with entity. This is an entity in the 2nd"
            " sentence."
        ),
        spans=[
            Span(text="entity", start=30, end=36, label="ENTITY"),
            Span(text="entity", start=49, end=55, label="ENTITY"),
        ],
    )

    ds = Dataset("test_dataset", data=[example])
    ds.apply_(
        "recon.augment.ent_label_sub.v1",
        label="ENTITY",
        subs=["new entity"],
        sub_prob=1.0,
    )

    expected_augmentation = Example(
        text=(
            "This is a first sentence with new entity. This is an new entity in the 2nd"
            " sentence."
        ),
        spans=[
            Span(text="new entity", start=30, end=40, label="ENTITY"),
            Span(text="new entity", start=53, end=63, label="ENTITY"),
        ],
    )

    assert len(ds) == 2
    assert example in ds.data
    assert expected_augmentation in ds.data
