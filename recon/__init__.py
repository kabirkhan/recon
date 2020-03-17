"""ReconNER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data."""

__version__ = "0.2.0"

from .corrections import *
from .corpus import Corpus
from .insights import *
from .loaders import read_json, read_jsonl
from .stats import get_ner_stats
from .validation import *
