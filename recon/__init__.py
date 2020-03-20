"""ReconNER, Debug annotated Named Entity Recognition (NER) data for inconsitencies and get insights on improving the quality of your data."""

__version__ = "0.2.0"

from .corpus import Corpus
from .corrections import *
from .insights import *
from .loaders import read_json, read_jsonl
from .stats import get_ner_stats
from .validation import *

try:
    # This needs to be imported in order for the entry points to be loaded
    from .prodigy import recipes as prodigy_recipes  # noqa: F401
except ImportError:
    pass
