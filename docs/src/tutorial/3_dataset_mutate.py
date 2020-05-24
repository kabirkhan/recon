from pathlib import Path

import typer
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def main(data_file: Path, output_file: Path):
    ds = Dataset("train").from_disk(data_file)

    print("STATS BEFORE")
    print("============")
    print(ds.apply(get_ner_stats, serialize=True))

    ds.apply_("recon.v1.upcase_labels")

    print("STATS AFTER")
    print("===========")
    print(ds.apply(get_ner_stats, serialize=True))


if __name__ == "__main__":
    typer.run(main)
