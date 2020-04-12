import shutil
from pathlib import Path

import pytest
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def test_dataset_initialize(example_data):

    dataset = Dataset("train")
