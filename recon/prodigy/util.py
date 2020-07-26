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

    examples = [e.dict() for e in examples]
    prodigy_examples = []
    for e in examples:
        prodigy_examples.append(set_hashes(e))

    db.add_examples(prodigy_examples, [prodigy_dataset])


def from_prodigy(prodigy_dataset: str) -> List[Example]:
    """Load Recon examples from a Prodigy dataset.

    Args:
        prodigy_dataset (str): Name of prodigy dataset to load from

    Returns:
        List[Example]: List of Recon examples
    """

    from prodigy.core import connect

    db = connect()

    if db.get_dataset(prodigy_dataset):
        return [Example(**e) for e in db.get_dataset(prodigy_dataset)]
