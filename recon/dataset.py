import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple, Union, cast

import srsly
from spacy.util import ensure_path
from wasabi import Printer

from .hashing import dataset_hash
from .loaders import read_json, read_jsonl
from .operations import registry
from .registry import loading_pipelines
from .store import ExampleStore
from .types import (
    DatasetOperationsState,
    Example,
    OperationResult,
    OperationState,
    OperationStatus,
)


class Dataset:
    """A Dataset is a around a List of examples.
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
        operations: List[OperationState] = None,
        example_store: ExampleStore = None,
        verbose: bool = False,
    ):
        self.name = name
        self.data = data
        if not operations:
            operations = []
        self.operations = operations

        if example_store is None:
            example_store = ExampleStore(data)
        self.example_store = example_store
        self.verbose = verbose

    @property
    def commit_hash(self) -> str:
        return cast(str, dataset_hash(self, as_int=False))

    def __hash__(self) -> int:
        return cast(int, dataset_hash(self))

    def __len__(self) -> int:
        return len(self.data)

    def apply(
        self, func: Callable[[List[Example], Any, Any], Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Apply a function to the dataset
        
        Args:
            func (Callable[[List[Example], Any, Any], Any]): 
                Function from an existing recon module that can operate on a List of examples
        
        Returns:
            Result of running func on List of examples
        """
        return func(self.data, *args, **kwargs)  # type: ignore

    def apply_(
        self,
        operation: Union[str, Callable[[Any], OperationResult]],
        *args: Any,
        initial_state: OperationState = None,
        **kwargs: Any,
    ) -> None:
        """Apply an operation to all data inplace.
        
        Args:
            operation (Callable[[Any], OperationResult]): Any operation that
                changes data in place. See recon.operations.registry.operations
        """
        if isinstance(operation, str):
            operation = registry.operations.get(operation)
            if operation:
                operation = cast(Callable, operation)

        name = getattr(operation, "name", None)
        if name is None or name not in registry.operations:
            raise ValueError(
                "This function is not an operation. Ensure your function is registered in the operations registry."
            )

        msg = Printer(no_print=self.verbose == False)
        msg.text(f"=> Applying operation '{name}' inplace")
        result: OperationResult = operation(self, *args, initial_state=initial_state, verbose=self.verbose, **kwargs)  # type: ignore
        msg.good(f"Completed operation '{name}'")

        self.operations.append(result.state)
        dataset_changed = any(
            (
                result.state.examples_added,
                result.state.examples_removed,
                result.state.examples_changed,
            )
        )
        if dataset_changed:
            self.data = result.data

    def pipe_(self, operations: List[Union[str, OperationState]]) -> None:
        """Run a sequence of operations on dataset data.
        Internally calls Dataset.apply_ and will resolve named
        operations in registry.operations
        
        Args:
            operations (List[Union[str, OperationState]]): List of operations
        """

        msg = Printer(no_print=self.verbose == False)
        msg.text(f"Applying pipeline of operations inplace to the dataset: {self.name}")

        for op in operations:
            op_name = op.name if isinstance(op, OperationState) else op
            msg.text(f"|_ {op_name}")

        for op in operations:
            if isinstance(op, str):
                op_name = op
                args = []
                kwargs = {}
                initial_state = None
            elif isinstance(op, OperationState):
                op_name = op.name
                args = op.args
                kwargs = op.kwargs
                initial_state = op

            operation = registry.operations.get(op_name)

            self.apply_(operation, *args, initial_state=initial_state, **kwargs)

    def from_disk(self, path: Path, loader_func: Callable = read_jsonl) -> "Dataset":
        """Load Dataset from disk given a path and a loader function that reads the data
        and returns an iterator of Examples
        
        Args:
            path (Path): path to load from
            loader_func (Callable, optional): Callable that reads a file and returns a List of examples. 
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        path = ensure_path(path)
        ds_op_state = None
        if (path.parent / ".recon" / self.name).exists():
            state = srsly.read_json(path.parent / ".recon" / self.name / "state.json")
            ds_op_state = DatasetOperationsState(**state)
            self.operations = ds_op_state.operations

        data = loader_func(path)
        self.data = data
        for example in self.data:
            self.example_store.add(example)

        if ds_op_state and self.commit_hash != ds_op_state.commit:
            # Dataset changed, examples added
            self.operations.append(
                OperationState(
                    name="examples_added_external",
                    status=OperationStatus.COMPLETED,
                    ts=datetime.now(),
                    examples_added=max(len(self) - ds_op_state.size, 0),
                    examples_removed=max(ds_op_state.size - len(self), 0),
                    examples_changed=0,
                    transformations=[],
                )
            )

            for op in self.operations:
                op.status = OperationStatus.NOT_STARTED

        seen: Set[str] = set()
        operations_to_run: Dict[str, OperationState] = {}

        for op in self.operations:
            if (
                op.name not in operations_to_run
                and op.name in registry.operations
                and op.status != OperationStatus.COMPLETED
            ):
                operations_to_run[op.name] = op

        for op_name, state in operations_to_run.items():
            op = registry.operations.get(op_name)
            self.apply_(op, *state.args, initial_state=state, **state.kwargs)  # type: ignore

        return self

    def to_disk(self, output_path: Path, force: bool = False, save_examples: bool = True) -> None:
        """Save Corpus to Disk
        
        Args:
            output_path (Path): Output file path to save data to
            force (bool): Force save to directory. Create parent directories
                or overwrite existing data.
            save_examples (bool): Save the example store along with the state.
        """
        output_path = ensure_path(output_path)
        output_dir = output_path.parent
        state_dir = output_dir / ".recon" / self.name
        if force:
            output_dir.mkdir(parents=True, exist_ok=True)

            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)

        ds_op_state = DatasetOperationsState(
            name=self.name, commit=self.commit_hash, size=len(self), operations=self.operations
        )
        srsly.write_json(state_dir / "state.json", ds_op_state.dict())

        if save_examples:
            self.example_store.to_disk(state_dir / "example_store.jsonl")

        srsly.write_jsonl(output_path, [e.dict() for e in self.data])
