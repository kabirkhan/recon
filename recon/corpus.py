from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import srsly
from spacy.util import ensure_path

from .dataset import Dataset
from .loaders import read_json, read_jsonl
from .store import ExampleStore
from .types import CorpusApplyResult, Example, OperationResult, OperationState


class Corpus:
    """Container for a full Corpus with train/dev/test splits.
    Used to apply core functions to all datasets at once.
    """

    def __init__(
        self,
        name: str,
        train: Dataset,
        dev: Dataset,
        test: Dataset = None,
        example_store: ExampleStore = None,
    ):
        """Initialize a Corpus.

        Args:
            name (str): Name of the Corpus
            train (Dataset): List of examples for **train** set
            dev (Dataset): List of examples for **dev** set
            test (Dataset, optional): Defaults to None. List of examples for **test** set
        """
        self._name = name
        if example_store is None:
            examples = train.data + dev.data
            if test:
                examples += test.data
            example_store = ExampleStore(examples)
        self.example_store = example_store

        if test is None:
            test = Dataset("test")

        for ds in (train, dev, test):
            ds.example_store = example_store

        self._train = train
        self._dev = dev
        self._test = test

    @property
    def name(self) -> str:
        """Get Corpus name

        Returns:
            str: Corpus name
        """
        return self._name

    @property
    def train_ds(self) -> Dataset:
        """Return train dataset

        Returns:
            Dataset: Train Dataset
        """
        return self._train

    @property
    def dev_ds(self) -> Dataset:
        """Return dev dataset

        Returns:
            Dataset: Dev Dataset
        """
        return self._dev

    @property
    def test_ds(self) -> Dataset:
        """Return test dataset

        Returns:
            Dataset: Test Dataset
        """
        return self._test

    @property
    def train(self) -> List[Example]:
        """Return train dataset

        Returns:
            List[Example]: Train Examples
        """
        return self._train.data

    @property
    def dev(self) -> List[Example]:
        """Return dev dataset

        Returns:
            List[Example]: Dev Examples
        """
        return self._dev.data

    @property
    def test(self) -> List[Example]:
        """Return test dataset

        Returns:
            List[Example]: Test Examples
        """
        return self._test.data or []

    @property
    def all(self) -> List[Example]:
        """Return concatenation of train/dev/test datasets

        Returns:
            List[Example]: All Examples in Corpus
        """
        return self.train + self.dev + self.test

    def apply(
        self, func: Callable[[List[Example], Any, Any], Any], *args: Any, **kwargs: Any
    ) -> CorpusApplyResult:
        """Apply a function to all datasets

        Args:
            func (Callable[[List[Example], Any, Any], Any]):
                Function from an existing recon module that can operate on a List of examples

        Returns:
            CorpusApplyResult: CorpusApplyResult mapping dataset name to return type of func Callable
        """

        return CorpusApplyResult(
            train=func(self.train, *args, **kwargs),  # type: ignore
            dev=func(self.dev, *args, **kwargs),  # type: ignore
            test=func(self.test, *args, **kwargs),  # type: ignore
            all=func(self.all, *args, **kwargs),  # type: ignore
        )

    def apply_(
        self, operation: Callable[[Any], OperationResult], *args: Any, **kwargs: Any
    ) -> None:
        """Apply a function to all data inplace.

        Args:
            operation (Callable[[Any], OperationResult]): Any operation that
                changes data in place. See recon.operations.registry.operations
        """
        self._train.apply_(operation, *args, **kwargs)
        self._dev.apply_(operation, *args, **kwargs)
        self._test.apply_(operation, *args, **kwargs)

    def pipe_(self, operations: List[Union[str, OperationState]]) -> None:
        """Run a sequence of operations on each dataset.
        Calls Dataset.pipe_ for each dataset

        Args:
            operations (List[Union[str, OperationState]]): List of operations
        """
        self._train.pipe_(operations)
        self._dev.pipe_(operations)
        self._test.pipe_(operations)

    def from_disk(
        self,
        data_dir: Path,
        train_file: str = "train.jsonl",
        dev_file: str = "dev.jsonl",
        test_file: str = "test.jsonl",
        loader_func: Callable = read_jsonl,
    ) -> "Corpus":
        """Load Corpus from disk given a directory with files
        named explicitly train.jsonl, dev.jsonl, and test.jsonl

        Args:
            data_dir (Path): directory to load from.
            train_file (str, optional): Filename of train data under data_dir. Defaults to train.jsonl.
            dev_file (str, optional): Filename of dev data under data_dir. Defaults to dev.jsonl.
            test_file (str, optional): Filename of test data under data_dir. Defaults to test.jsonl.
            loader_func (Callable, optional): Callable that reads a file and returns a List of examples.
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        data_dir = ensure_path(data_dir) / self.name

        train = Dataset("train").from_disk(data_dir / train_file)
        dev = Dataset("dev").from_disk(data_dir / dev_file)

        try:
            test = Dataset("test").from_disk(data_dir / test_file)
            corpus = self(self.name, train, dev, test=test)
        except ValueError as e:
            corpus = self(self.name, train, dev)
        return corpus

    def to_disk(self, data_dir: Path, force: bool = False) -> None:
        """Save Corpus to Disk

        Args:
            data_dir (Path): Directory to save data to
            force (bool): Force save to directory. Create parent directories
                or overwrite existing data.
        """
        data_dir = ensure_path(data_dir) / self.name
        state_dir = data_dir / ".recon"
        if force:
            data_dir.mkdir(parents=True, exist_ok=True)

            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)

        self._train.to_disk(data_dir / "train.jsonl", force=force, save_examples=False)
        self._dev.to_disk(data_dir / "dev.jsonl", force=force, save_examples=False)
        if self._test:
            self._test.to_disk(data_dir / "test.jsonl", force=force, save_examples=False)

        self.example_store.to_disk(state_dir / "example_store.jsonl")

    def from_prodigy(
        self,
        prodigy_train_datasets: List[str] = None,
        prodigy_dev_datasets: List[str] = None,
        prodigy_test_datasets: List[str] = None,
    ) -> "Corpus":
        """Load a Corpus from 3 separate datasets in Prodigy

        Args:
            prodigy_train_dataset (str, optional): Train dataset name in Prodigy
            prodigy_dev_dataset (str, optional): Dev dataset name in Prodigy
            prodigy_test_dataset (str, optional): Test dataset name in Prodigy

        Returns:
            Corpus: Corpus initialized from prodigy datasets
        """
        train_ds = Dataset("train").from_prodigy(prodigy_train_datasets)
        dev_ds = Dataset("dev").from_prodigy(prodigy_dev_datasets)
        test_ds = (
            Dataset("test").from_prodigy(prodigy_train_datasets) if prodigy_test_dataset else None
        )

        ds = self(self.name, train_ds, dev_ds, test_ds)
        return ds

    def to_prodigy(
        self,
        prodigy_train_dataset: str = None,
        prodigy_dev_dataset: str = None,
        prodigy_test_dataset: str = None,
    ):
        """Save a Corpus to 3 separate Prodigy datasets

        Args:
            prodigy_train_dataset (str, optional): Train dataset name in Prodigy
            prodigy_dev_dataset (str, optional): Dev dataset name in Prodigy
            prodigy_test_dataset (str, optional): Test dataset name in Prodigy
        """
        if not prodigy_train_dataset:
            prodigy_train_dataset = f"{self.name}_train_{self.train.commit_hash}"

        if not prodigy_dev_dataset:
            prodigy_dev_dataset = f"{self.name}_dev_{self.dev.commit_hash}"

        if not prodigy_test_dataset:
            prodigy_test_dataset = f"{self.name}_test_{self.test.commit_hash}"

        self.train_ds.to_prodigy(prodigy_train_dataset)
        self.dev_ds.to_prodigy(prodigy_dev_dataset)
        self.test_ds.to_prodigy(prodigy_test_dataset)
