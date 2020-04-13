from pathlib import Path

import srsly
import typer
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def main(data_dir: Path):
    ds = Dataset.from_disk(data_dir)
    ds_stats = ds.apply(get_ner_stats, serialize=True, no_print=True)
    for name, stats in ds_stats.items():
        print(f"{name}")
        print("=" * 50)
        print(stats)


if __name__ == "__main__":
    typer.run(main)
