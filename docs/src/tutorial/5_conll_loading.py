from datasets.load import load_dataset
from recon import Corpus, Dataset


def main():

    # Download the raw data using HF Datasets
    conll2003 = load_dataset("conll2003")

    # Define the str BIO labels based on the numerical tags stored in HF Datasets
    conll_labels = [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
        "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ]

    # Use the Dataset.from_hf_dataset method to make an
    # Example from each row and create Recon Datasets for
    # each split in the raw data
    train_ds = Dataset("train").from_hf_dataset(
        conll2003["train"], labels=conll_labels
    )
    dev_ds = Dataset("dev").from_hf_dataset(
        conll2003["validation"], labels=conll_labels
    )
    test_ds = Dataset("test").from_hf_dataset(
        conll2003["test"], labels=conll_labels
    )

    # Initialize a Recon Corpus from the 3 Datasets
    corpus = Corpus("conll2003", train_ds, dev_ds, test_ds)
    print(corpus)

    corpus.to_disk("./examples/data/conll2003", overwrite=True)


if __name__ == "__main__":
    main()
