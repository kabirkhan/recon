from pathlib import Path
from typing import Any, Callable, Dict, List
from reconner.loaders import read_jsonl, read_json
from reconner.types import Example


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
    
    def __init__(self,
                 train: List[Example],
                 dev: List[Example],
                 test: List[Example] = None):
        self.datasets = {
            'train': train,
            'dev': dev,
            'all': train + dev
        }
        if test:
            self.datasets.update({
                'test': test,
                'all': train + dev + test
            })
    
    @classmethod
    def from_disk(cls, path: Path, loader_func: Callable = read_jsonl):
        """Load Dataset from disk given a directory with files 
        named explicitly train.jsonl, dev.jsonl, and test.jsonl
        
        ### Parameters
        --------------
        **path**: (Path), required.
            directory to load from
        **loader_func**: (Callable, optional), Defaults to read_jsonl.
            Loader function (TODO: Make this a bit more generic)
        """
        return Dataset(
            loader_func(path / 'train.jsonl'),
            loader_func(path / 'dev.jsonl'),
            test=loader_func(path / 'test.jsonl')
        )

    def apply(self,
              func: Callable[[List[Example]], Any],
              *args: Any,
              **kwargs: Any) -> Dict[str, List[Example]]:
        """Apply an existing function to all datasets
        
        ### Parameters
        --------------
        **func**: (Callable[[List[Example]], Any]), required.
            Function from an existing reconner module that can operate on a List of Examples
        
        ### Returns
        -----------
        (Dict[str, List[Example]]): 
            Dictionary mapping dataset names to List[Example], same as the internal datasets property
        """
        res = {}
        for k, dataset in self.datasets.items():
            res[k] = func(dataset, *args, **kwargs)
        return res
