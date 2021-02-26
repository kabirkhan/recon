from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import srsly
from recon.dataset import Dataset
from recon.loaders import read_jsonl
from recon.store import ExampleStore
from recon.types import (
    CorpusApplyResult,
    CorpusMeta,
    Example,
    OperationResult,
    OperationState,
)
from spacy.util import ensure_path


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

    @classmethod
    def from_disk(
        cls,
        data_dir: Path,
        name: str = "corpus",
        train_name: str = "train",
        dev_name: str = "dev",
        test_name: str = "test",
        loader_func: Callable = read_jsonl,
    ) -> "Corpus":
        """Load Corpus from disk given a directory with files
        named explicitly train.jsonl, dev.jsonl, and test.jsonl

        Args:
            data_dir (Path): directory to load from.
            train_name (str, optional): Name of train data under data_dir. Defaults to train.
            dev_name (str, optional): Name of dev data under data_dir. Defaults to dev.
            test_name (str, optional): Name of test data under data_dir. Defaults to test.
            loader_func (Callable, optional): Callable that reads a file and returns a List of examples.
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        data_dir = ensure_path(data_dir)

        corpus_meta_path = data_dir / ".recon" / "meta.json"
        if corpus_meta_path.exists():
            corpus_meta = CorpusMeta.parse_file(corpus_meta_path)
            name = corpus_meta.name

        example_store_path = data_dir / ".recon" / "example_store.jsonl"
        example_store = ExampleStore()
        if example_store_path.exists():
            example_store.from_disk(example_store_path)

        train = Dataset("train", example_store=example_store).from_disk(data_dir)
        dev = Dataset("dev", example_store=example_store).from_disk(data_dir)

        try:
            test = Dataset("test", example_store=example_store).from_disk(data_dir)
            corpus = cls(name, train, dev, test=test)
        except ValueError:
            corpus = cls(name, train, dev)
        return corpus

    def to_disk(self, data_dir: Path, force: bool = False) -> None:
        """Save Corpus to Disk

        Args:
            data_dir (Path): Directory to save data to
            force (bool): Force save to directory. Create parent directories
                or overwrite existing data.
        """
        data_dir = ensure_path(data_dir)
        state_dir = data_dir / ".recon"
        corpus_meta_path = state_dir / "meta.json"

        if force:
            data_dir.mkdir(parents=True, exist_ok=True)

            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)

        srsly.write_json(corpus_meta_path, CorpusMeta(name=self.name).dict())
        self._train.to_disk(data_dir, force=force, save_examples=False)
        self._dev.to_disk(data_dir, force=force, save_examples=False)
        if self._test:
            self._test.to_disk(data_dir, force=force, save_examples=False)

        self.example_store.to_disk(state_dir / "example_store.jsonl")

    @classmethod
    def from_prodigy(
        cls,
        name: str,
        prodigy_train_datasets: List[str],
        prodigy_dev_datasets: List[str],
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
            Dataset("test").from_prodigy(prodigy_test_datasets) if prodigy_test_datasets else None
        )

        ds = cls(name, train_ds, dev_ds, test_ds)
        return ds

    def to_prodigy(
        self,
        name: str = None,
        prodigy_train_dataset: str = None,
        prodigy_dev_dataset: str = None,
        prodigy_test_dataset: str = None,
        overwrite: bool = True,
    ) -> Tuple[str, str, str]:
        """Save a Corpus to 3 separate Prodigy datasets

        Args:
            prodigy_train_dataset (str, optional): Train dataset name in Prodigy
            prodigy_dev_dataset (str, optional): Dev dataset name in Prodigy
            prodigy_test_dataset (str, optional): Test dataset name in Prodigy
        """
        name = name if name else self.name

        if not prodigy_train_dataset:
            prodigy_train_dataset = f"{name}_train_{self.train_ds.commit_hash}"

        if not prodigy_dev_dataset:
            prodigy_dev_dataset = f"{name}_dev_{self.dev_ds.commit_hash}"

        if not prodigy_test_dataset:
            prodigy_test_dataset = f"{name}_test_{self.test_ds.commit_hash}"

        self.train_ds.to_prodigy(prodigy_train_dataset, overwrite=overwrite)
        self.dev_ds.to_prodigy(prodigy_dev_dataset, overwrite=overwrite)
        self.test_ds.to_prodigy(prodigy_test_dataset, overwrite=overwrite)

        return (prodigy_train_dataset, prodigy_dev_dataset, prodigy_test_dataset)
