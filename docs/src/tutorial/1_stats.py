from pathlib import Path

import typer
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def main(data_dir: Path):
    ds = Dataset.from_disk(data_dir)
    train_stats = get_ner_stats(ds.train)
    get_ner_stats(ds.train, serialize=True)


if __name__ == "__main__":
    typer.run(main)
