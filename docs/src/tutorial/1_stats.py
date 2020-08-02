from pathlib import Path

import typer
from recon.loaders import read_jsonl
from recon.stats import get_ner_stats


def main(data_file: Path):
    data = read_jsonl(data_file)
    get_ner_stats(data)
    print(get_ner_stats(data, serialize=True))


if __name__ == "__main__":
    typer.run(main)
