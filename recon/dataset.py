from pathlib import Path
from typing import Any, Callable, Dict, List

from spacy.util import ensure_path

from .loaders import read_json, read_jsonl
from .types import Example


class Dataset:
    """Container for a full dataset with train/dev/test splits.
    Used to apply core functions to all datasets at once.

    ### Parameters
    --------------
    **train**: (List[Example]), required.
        List of Examples for **train** set
    **dev**: (List[Example]), required.
        List of Examples for **dev** set
    **test**: (List[Example], optional), Defaults to None.
        List of Examples for **test** set
    """

    def __init__(
        self, train: List[Example], dev: List[Example], test: List[Example] = None
    ):
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
    ) -> "Dataset":
        """Load Dataset from disk given a directory with files 
        named explicitly train.jsonl, dev.jsonl, and test.jsonl
        
        ### Parameters
        --------------
        **data_dir**: (Path), required.
            directory to load from
        **train_file**: (str, optional), Defaults to train.jsonl.
            Filename of train data under path
        **dev_file**: (str, optional), Defaults to dev.jsonl.
            Filename of dev data under path
        **test_file**: (str, optional), Defaults to test.jsonl.
            Filename of test data under path
        **loader_func**: (Callable, optional), Defaults to read_jsonl.
            Loader function (TODO: Make this a bit more generic)
        """

        data_dir = ensure_path(data_dir)

        train_data = loader_func(data_dir / train_file)
        dev_data = loader_func(data_dir / dev_file)

        try:
            test_data = loader_func(data_dir / test_file)
            ds = Dataset(train_data, dev_data, test=test_data)
        except ValueError as e:
            ds = Dataset(train_data, dev_data)
        return ds

    @property
    def train(self) -> List[Example]:
        return self._train

    @property
    def dev(self) -> List[Example]:
        return self._dev

    @property
    def test(self) -> List[Example]:
        return self._test or []

    @property
    def all(self) -> List[Example]:
        return self.train + self.dev + self.test

    def apply(
        self, func: Callable[[List[Example], Any, Any], Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Apply an existing function to all datasets
        
        ### Parameters
        --------------
        **func**: (Callable[[List[Example]], Any]), required.
            Function from an existing recon module that can operate on a List of Examples
        
        ### Returns
        -----------
        (Dict[str, Any): 
            Dictionary mapping dataset names to return type of func Callable
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
        assert isinstance(new_data[0], Example)
        return new_data

    def apply_(
        self,
        func: Callable[[List[Example], Any, Any], List[Example]],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Apply an existing function to all data inplace
        
        ### Parameters
        --------------
        **func**: (Callable[[List[Example]], List[Example]]), required.
            Function from an existing recon module that can operate on a List of Examples
        """

        new_train = self._validate_inplace_apply(self.train, func(self.train, *args, **kwargs))  # type: ignore
        new_dev = self._validate_inplace_apply(self.dev, func(self.dev, *args, **kwargs))  # type: ignore
        new_test = self._validate_inplace_apply(self.test, func(self.test, *args, **kwargs))  # type: ignore

        self._train = new_train
        self._dev = new_dev
        self._test = new_test
