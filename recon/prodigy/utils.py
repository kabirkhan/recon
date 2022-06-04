from typing import List

from recon.types import Example


def to_prodigy(
    examples: List[Example],
    prodigy_dataset: str,
    overwrite_dataset: bool = False,
    add_hash: bool = True,
) -> None:
    """Save a list of examples to Prodigy

    Args:
        examples (List[Example]): Input examples
        prodigy_dataset (str): Name of Prodigy dataset to save to
        overwrite_dataset (bool, optional): If True will overwrite all data in Prodigy

    Raises:
        ValueError: If trying to save examples to an existing dataset without explicitly
            setting overwrite_dataset to True
    """
    from prodigy.core import connect
    from prodigy.util import set_hashes

    db = connect()

    if db.get_dataset(prodigy_dataset):
        if overwrite_dataset:
            db.drop_dataset(prodigy_dataset)
            db.add_dataset(prodigy_dataset)
        else:
            raise ValueError(f"Prodigy dataset {prodigy_dataset} already exists.")
    else:
        db.add_dataset(prodigy_dataset)

    prodigy_examples = []
    for e in examples:
        prodigy_examples.append(set_hashes(e.dict(exclude_unset=True)))

    db.add_examples(prodigy_examples, [prodigy_dataset])


def from_prodigy(prodigy_dataset: str) -> List[Example]:
    """Load Recon examples from a Prodigy dataset.

    Args:
        prodigy_dataset (str): Name of prodigy dataset to load from

    Raises:
        ValueError: If trying to load examples from a dataset that doesn't exist in prodigy

    Returns:
        List[Example]: List of Recon examples
    """

    from prodigy.core import connect

    db = connect()

    examples = []
    prodigy_examples = db.get_dataset(prodigy_dataset)
    if not prodigy_examples:
        raise ValueError(
            f"Prodigy dataset with name {prodigy_dataset} does not exist. Available datasets are: \n {', '.join(db.datasets)}"
        )
    examples = [Example(**e) for e in prodigy_examples]
    return examples
