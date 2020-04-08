from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import srsly
from spacy.util import ensure_path

from .hashing import dataset_hash
from .loaders import read_json, read_jsonl
from .registry import loading_pipelines
from .store import ExampleStore
from .types import DatasetOperationsState, Example, OperationResult, OperationState, OperationStatus


class Dataset:
    """A Dataset is a around a List of Examples.
    Datasets are responsible for tracking all Operations done on them.
    This ensures data lineage and easy reporting of how changes in the 
    data based on various Operations effects overall quality.
    
    Dataset holds state (let's call it self.operations for now)
    self.operations is a list of every function run on the Dataset since it's
    initial creation. If loading from disk, track everything that happens in loading
    phase in operations as well by simply initializing self.operations in constructors
    
    Each operation should has the following attributes:
        operation hash
        name: function/callable name ideally, could be added with a decorator
        status: (not_started|completed)
        transformations: List[Transformation]
            commit hash
            timestamp(s) - start and end both? end is probably enough
            examples deleted
            examples added
            examples corrected
            annotations deleted
            annotations added
            annotations corrected

            for annotations deleted/added/corrected, include mapping from old Example hash to new Example hash
            that can be decoded for display later

    All operations are serializable in the to_disk and from_disk methods.

    So if I have 10 possible transformations.

    I can run 1..5, save to disk train a model and check results. 
    Then I can load that model from disk with all previous operations already tracked
    in self.operations. Then I can run 6..10, save to disk and train model.
    Now I have git-like "commits" for the data used in each model.
    
    """

    def __init__(
        self,
        name: str,
        data: List[Example] = [],
        global_state: Dict[str, List[OperationState]] = {},
        operations: List[OperationState] = [],
        iteration: int = 0,
        example_store: ExampleStore = None
    ):
        self.name = name
        self.data = data
        self.global_state = global_state
        self.operations = operations
        self.iteration = iteration

        if example_store is None:
            example_store = ExampleStore(data)
        self.example_store = example_store
    
    @property
    def commit_hash(self):
        return dataset_hash(self, as_int=False)

    def __hash__(self):
        return dataset_hash(self)
    
    def apply(
        self, func: Callable[[List[Example], Any, Any], Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Apply a function to the dataset
        
        Args:
            func (Callable[[List[Example], Any, Any], Any]): 
                Function from an existing recon module that can operate on a List of Examples
        
        Returns:
            Result of running func on List of Examples
        """
        return func(self.data, *args, **kwargs) # type: ignore
        
    def apply_(
        self, operation: Callable[[Any], OperationResult], *args: Any, **kwargs: Any
    ) -> None:
        """Apply a function to all data inplace.
        
        Args:
            func (Callable[[List[Example], Any, Any], List[Example]]): Function from an existing recon module
                that can operate on a List[Example] and return a List[Example]
        """
        result: OperationResult = operation(self.data, *args, **kwargs)  # type: ignore
        dataset_changed = any((result.state.examples_added, result.state.examples_removed, result.state.examples_changed))
        if dataset_changed:
            self.iteration += 1

            self.global_state[result.state.name] = result.state
            self.operations.append(result.state)
            self.data = result.data

    def from_disk(
        self,
        path: Path,
        loader_func: Callable = read_jsonl,
        loading_pipeline: List[str] = loading_pipelines.get("default")()
    ) -> "Dataset":
        """Load Dataset from disk given a path and a loader function that reads the data
        and returns an iterator of Examples
        
        Args:
            path (Path): path to load from
            loader_func (Callable, optional): Callable that reads a file and returns a List of Examples. 
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        path = ensure_path(path)
        if (path / ".recon" / self.name).exists():
            cfg = srsly.read_json(path / ".recon" / self.name)
            print(cfg)
            self.operations = [OperationState(**op) for op in cfg["operations"]]

        idx_to_remove: Set[int] = set()
        for idx, op in enumerate(self.operations):
            if op.name in loading_pipeline and op.status == OperationStatus.COMPLETED:
                idx_to_remove.add(idx)

        loading_pipeline = [op for i, op in enumerate(loading_pipeline) if i not in idx_to_remove]

        data = loader_func(path, loading_pipeline=loading_pipeline)
        self.data = data

        return self

    def to_disk(self, output_path: Path, force: bool = False) -> None:
        """Save Corpus to Disk
        
        Args:
            output_path (Path): Output file path to save data to
            force (bool): Force save to directory. Create parent directories
                or overwrite existing data.
        """
        output_path = ensure_path(output_path)
        output_dir = output_path.parent
        cfg_dir = output_dir / ".recon" / self.name
        if force:
            output_dir.mkdir(parents=True, exist_ok=True)
        
            if not cfg_dir.exists():
                cfg_dir.mkdir(parents=True, exist_ok=True)

        cfg = DatasetOperationsState(name=self.name, operations=self.operations)
        with (cfg_dir / "state.json").open("w+") as cfg_f:
            cfg_f.write(cfg.json())

        srsly.write_jsonl(output_path, [e.dict() for e in self.data])
