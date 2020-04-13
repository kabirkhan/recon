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
        self, train: Dataset, dev: Dataset, test: Dataset = None, example_store: ExampleStore = None
    ):
        """Initialize a Corpus.
        
        Args:
            train (Dataset): List of examples for **train** set
            dev (Dataset): List of examples for **dev** set
            test (Dataset, optional): Defaults to None. List of examples for **test** set
        """
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
    def train(self) -> List[Example]:
        """Return train dataset
        
        Returns:
            List[Example]: Train Examples
        """
        return self._train.data

    @property
    def dev(self) -> List[Example]:
        """Return train dev
        
        Returns:
            List[Example]: Train Examples
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
        data_dir = ensure_path(data_dir)

        train = Dataset("train").from_disk(data_dir / train_file)
        dev = Dataset("dev").from_disk(data_dir / dev_file)

        try:
            test = Dataset("test").from_disk(data_dir / test_file)
            corpus = cls(train, dev, test=test)
        except ValueError as e:
            corpus = cls(train, dev)
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
        if force:
            data_dir.mkdir(parents=True, exist_ok=True)

            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)

        self._train.to_disk(data_dir / "train.jsonl", force=force, save_examples=False)
        self._dev.to_disk(data_dir / "dev.jsonl", force=force, save_examples=False)
        if self._test:
            self._test.to_disk(data_dir / "test.jsonl", force=force, save_examples=False)

        self.example_store.to_disk(state_dir / "example_store.jsonl")
