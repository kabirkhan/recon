from pathlib import Path
from typing import Callable

from recon.corpus import Corpus
from recon.stats import (
    calculate_label_balance_entropy,
    calculate_label_distribution_similarity,
    get_ner_stats,
)
from recon.types import Example, NERStats
from wasabi import Printer


def stats(data_dir: Path) -> None:
    """Calculate statistics on a Corpus

    Args:
        data_dir (Path): Path to data folder
    """

    msg: Printer = Printer()

    def print_stats(corpus: Corpus) -> None:
        ner_stats: NERStats
        for ds, ner_stats in corpus.apply(get_ner_stats).items():
            sorted_labels = sorted(ner_stats.n_annotations_per_type.keys())

            msg.text(f"Stats for {ds.capitalize()} data")
            msg.text("--------------------")
            msg.table(
                {
                    "N Examples": ner_stats.n_examples,
                    "N Annotations": ner_stats.n_annotations,
                }
            )
            msg.info(f"Labels in {ds}")
            msg.text(sorted_labels)

            msg.info("N Annotations per Label")
            msg.table(ner_stats.n_annotations_per_type)

    with msg.loading("Loading Corpus from Disk"):
        corpus = Corpus.from_disk(data_dir)
        msg.good("Done")

    msg.divider("Calculating stats")

    print_stats(corpus)
    train_dev_sim = calculate_label_distribution_similarity(corpus.train, corpus.dev)
    train_test_sim = calculate_label_distribution_similarity(corpus.train, corpus.test)
    dev_test_sim = calculate_label_distribution_similarity(corpus.dev, corpus.test)

    msg.divider("Calculating Similarity between label distributions of Corpus")

    msg.table(
        {
            "Train / Dev": round(train_dev_sim, 2),
            "Train / Test": round(train_test_sim, 2),
            "Dev   / Test": round(dev_test_sim, 2),
        }
    )
