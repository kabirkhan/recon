"""ReconNER, Debug annotated Named Entity Recognition (NER) data for
inconsitencies and get insights on improving the quality of your data."""

__version__ = "0.14.1"

from recon.corpus import Corpus
from recon.dataset import Dataset
from recon.types import Example

try:
    # This needs to be imported in order for the entry points to be loaded
    from recon import prodigy # noqa
except ImportError:
    pass


__all__ = ["Corpus", "Dataset", "Example"]
