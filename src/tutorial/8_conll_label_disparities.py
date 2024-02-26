from pprint import pprint

from recon import Corpus
from recon.insights import get_label_disparities


def main():
    # Load the Conll Corpus
    corpus = Corpus.from_disk("./examples/data/conll2003", "conll2003")

    test_ld = get_label_disparities(corpus.test, "LOC", "PER")
    pprint(test_ld)


if __name__ == "__main__":
    main()
