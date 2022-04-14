"""ReconNER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data."""

__version__ = "0.9.2"

from recon.augmentation import *
from recon.corpus import Corpus
from recon.dataset import Dataset
from recon.insights import *
from recon.loaders import read_json, read_jsonl
from recon.operations.core import operation
from recon.operations.corrections import *
from recon.operations.tokenization import add_tokens, fix_tokenization_and_spacing
from recon.operations.validation import filter_overlaps, upcase_labels
from recon.preprocess import SpacyPreProcessor
from recon.stats import *

try:
    # This needs to be imported in order for the entry points to be loaded
    from recon.prodigy import recipes as prodigy_recipes  # noqa: F401
except ImportError as e:
    pass
