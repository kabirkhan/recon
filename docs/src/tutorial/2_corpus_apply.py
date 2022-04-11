from pathlib import Path

import typer
from recon import Corpus


def main(data_dir: Path):
    corpus = Corpus.from_disk(data_dir)
    res = corpus.apply("recon.v1.get_ner_stats", serialize=True, no_print=True)
    for name, stats in res.items():
        print(f"{name}")
        print("=" * 50)
        print(stats)


if __name__ == "__main__":
    typer.run(main)
