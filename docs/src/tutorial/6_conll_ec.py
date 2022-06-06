from pprint import pprint

from recon import Corpus
from recon.stats import get_entity_coverage, calculate_entity_coverage_entropy, calculate_entity_coverage_similarity


def main():

    # Load the Conll Corpus
    corpus = Corpus.from_disk("./examples/data/conll2003", "conll2003")

    train_entity_cov = get_entity_coverage(corpus.train, case_sensitive=True)

    print("Most Covered Entities")
    pprint(train_entity_cov[:10])
    print()
    print("Least Covered Entities")
    pprint(train_entity_cov[-10:])


if __name__ == "__main__":
    main()
