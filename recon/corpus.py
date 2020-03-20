from pathlib import Path
from typing import Any, Callable, Dict, List

import srsly
from spacy.util import ensure_path

from .loaders import read_json, read_jsonl
from .types import Example


class Corpus:
    """Container for a full Corpus with train/dev/test splits.
    Used to apply core functions to all datasets at once.
    """

    def __init__(
        self, train: List[Example], dev: List[Example], test: List[Example] = None
    ):
        """Initialize a Corpus.
        
        Args:
            train (List[Example]): List of Examples for **train** set
            dev (List[Example]): List of Examples for **dev** set
            test (List[Example], optional): Defaults to None. List of Examples for **test** set
        """
        self._train = train
        self._dev = dev
        self._test = test

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
            loader_func (Callable, optional): Callable that reads a file and returns a List of Examples. 
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        data_dir = ensure_path(data_dir)

        train_data = loader_func(data_dir / train_file)
        dev_data = loader_func(data_dir / dev_file)

        try:
            test_data = loader_func(data_dir / test_file)
            corpus = cls(train_data, dev_data, test=test_data)
        except ValueError as e:
            corpus = cls(train_data, dev_data)
        return corpus

    def to_disk(self, data_dir: Path, force: bool = False) -> None:
        """Save Corpus to Disk
        
        Args:
            data_dir (Path): Directory to save data to
            force (bool): Force save to directory. Create parent directories
                or overwrite existing data.
        """
        data_dir = ensure_path(data_dir)
        if force:
            data_dir.mkdir(parents=True, exist_ok=True)

        def serialize(examples: List[Example]) -> List[Dict[str, object]]:
            return [e.dict() for e in examples]

        srsly.write_jsonl(data_dir / "train.jsonl", serialize(self.train))
        srsly.write_jsonl(data_dir / "dev.jsonl", serialize(self.dev))
        srsly.write_jsonl(data_dir / "test.jsonl", serialize(self.test))

    @property
    def train(self) -> List[Example]:
        """Return train dataset
        
        Returns:
            List[Example]: Train Examples
        """
        return self._train

    @property
    def dev(self) -> List[Example]:
        """Return train dev
        
        Returns:
            List[Example]: Train Examples
        """
        return self._dev

    @property
    def test(self) -> List[Example]:
        """Return test dataset
        
        Returns:
            List[Example]: Test Examples
        """
        return self._test or []

    @property
    def all(self) -> List[Example]:
        """Return concatenation of train/dev/test datasets
        
        Returns:
            List[Example]: All Examples in Corpus
        """
        return self.train + self.dev + self.test

    def apply(
        self, func: Callable[[List[Example], Any, Any], Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Apply a function to all datasets
        
        Args:
            func (Callable[[List[Example], Any, Any], Any]): 
                Function from an existing recon module that can operate on a List of Examples
        
        Returns:
            Dict[str, Any]: Dictionary mapping dataset names to return type of func Callable
        """
        res = {
            "train": func(self.train, *args, **kwargs),  # type: ignore
            "dev": func(self.dev, *args, **kwargs),  # type: ignore
            "test": func(self.test, *args, **kwargs),  # type: ignore
            "all": func(self.all, *args, **kwargs),  # type: ignore
        }

        return res

    def _validate_inplace_apply(
        self, old_data: List[Example], new_data: List[Example]
    ) -> List[Example]:
        assert isinstance(new_data, list)
        assert len(new_data) == len(old_data)
        assert new_data[0].text == old_data[0].text
        return new_data

    def apply_(
        self, func: Callable[[Any], List[Example]], *args: Any, **kwargs: Any
    ) -> None:
        """Apply a function to all data inplace.
        
        Args:
            func (Callable[[List[Example], Any, Any], List[Example]]): Function from an existing recon module
                that can operate on a List[Example] and return a List[Example]
        """
        new_train = self._validate_inplace_apply(self.train, func(self.train, *args, **kwargs))  # type: ignore
        new_dev = self._validate_inplace_apply(self.dev, func(self.dev, *args, **kwargs))  # type: ignore
        new_test = self._validate_inplace_apply(self.test, func(self.test, *args, **kwargs))  # type: ignore

        self._train = new_train
        self._dev = new_dev
        self._test = new_test
