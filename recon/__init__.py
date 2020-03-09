"""ReconNER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data."""

__version__ = "0.1.2"

from .corrections import *
from .dataset import Dataset
from .insights import *
from .loaders import read_json, read_jsonl
from .stats import ner_stats
from .validation import *
