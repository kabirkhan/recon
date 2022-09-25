from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import spacy
import srsly
from spacy.tokens import Doc
from wasabi import Printer

from recon.hashing import dataset_hash
from recon.loaders import from_spacy, read_jsonl, to_spacy
from recon.operations import registry
from recon.stats import get_ner_stats
from recon.store import ExampleStore
from recon.types import (
    ApplyType,
    DatasetOperationsState,
    Example,
    OperationResult,
    OperationState,
    OperationStatus,
    Span,
    Stats,
    Token,
    TransformationType,
)
from recon.utils import ensure_path

if TYPE_CHECKING:
    try:
        from datasets.arrow_dataset import Dataset as HFDataset
    except ImportError:
        pass


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
        operations: List[OperationState] = [],
        example_store: Optional[ExampleStore] = None,
        verbose: bool = True,
    ):
        self._name = name
        self._data = data
        if not operations:
            operations = []
        self._operations = operations

        if example_store is None:
            example_store = ExampleStore(data)
        self._example_store = example_store
        self._verbose = verbose
        self._stats: Optional[Stats] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def commit_hash(self) -> str:
        """String representation of internal hash
        that can be used to mark a checkpoint in a dataset."""
        return cast(str, dataset_hash(self, as_int=False))

    @property
    def data(self) -> List[Example]:
        return self._data

    @property
    def operations(self) -> List[OperationState]:
        return self._operations

    @property
    def example_store(self) -> ExampleStore:
        return self._example_store

    @property
    def stats(self) -> Stats:
        return get_ner_stats(self.data)

    def summary(self) -> str:
        return f"Dataset\nName: {self.name}\nStats: {self.stats}"

    def print_summary(self) -> None:
        print(self.summary())

    def __str__(self) -> str:
        return self.summary()

    def __hash__(self) -> int:
        return cast(int, dataset_hash(self))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, example_hash: int) -> Example:
        for e in self.data:
            if hash(e) == example_hash:
                return e
        raise KeyError(f"Example with hash {example_hash} does not exist")

    def apply(self, func: ApplyType, *args: Any, **kwargs: Any) -> Any:
        """Apply a function to the dataset

        Args:
            func (Callable[[List[Example], Any], Any]):
                Function from an existing recon module that can operate on a List of examples

        Returns:
            Result of running func on List of examples
        """
        return func(self.data, *args, **kwargs)  # type: ignore

    def apply_(
        self,
        operation: Union[str, Callable[[Any], OperationResult]],
        *args: Any,
        initial_state: Optional[OperationState] = None,
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
        if not name:
            raise ValueError("This function is not an operation since it does not have a name.")

        msg = Printer(no_print=not self._verbose)
        msg.text(f"=> Applying operation '{name}' to dataset '{self.name}'")
        result: OperationResult = operation(self, *args, initial_state=initial_state, verbose=self._verbose, **kwargs)  # type: ignore
        msg.good(f"Completed operation '{name}'")

        self._operations.append(result.state)
        dataset_changed = any(
            (
                result.state.examples_added,
                result.state.examples_removed,
                result.state.examples_changed,
            )
        )
        if dataset_changed:
            self._data = result.data

    def pipe_(self, operations: List[Union[str, OperationState]]) -> None:
        """Run a sequence of operations on dataset data.
        Internally calls Dataset.apply_ and will resolve named
        operations in registry.operations

        Args:
            operations (List[Union[str, OperationState]]): List of operations
        """

        msg = Printer(no_print=not self._verbose)
        msg.text(f"Applying pipeline of operations inplace to Dataset: {self.name}")

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
            else:
                raise ValueError(
                    "Operation is not resolvable. Must be a name for a registered operation or an instance of OperationState."
                )

            operation = registry.operations.get(op_name)
            self.apply_(operation, *args, initial_state=initial_state, **kwargs)

    def rollback(self, n: int = 1) -> None:
        """Rollback the last n operations on a dataset.

        e.g.
            ```
            ds = Dataset("name", data)

            initial_ds_hash = hash(ds)

            ds.apply_("some_operation")
            ds.rollback()

            hash(ds) == initial_ds_hash
            >>> True # This should be True

        Args:
            n (int): Number of operations to rollback
        """

        if n < 1:
            raise ValueError(f"Cannot rollback dataset: provided n: ({n}) must be 1 or higher.")
        elif n > len(self.operations):
            raise ValueError(
                f"Cannot rollback dataset: provided n ({n}) is larger than the total number of dataset operations."
            )

        store = self.example_store
        examples_to_remove = set()
        examples_to_add = []

        for op in self.operations[-n:]:
            for t in op.transformations:
                if t.type == TransformationType.EXAMPLE_ADDED:
                    examples_to_remove.add(t.example)
                elif t.type == TransformationType.EXAMPLE_CHANGED:
                    examples_to_remove.add(t.example)
                    examples_to_add.append(store[t.prev_example])  # type: ignore
                elif t.type == TransformationType.EXAMPLE_REMOVED:
                    examples_to_add.append(store[t.prev_example])  # type: ignore

        old_data = [e for e in self.data if hash(e) not in examples_to_remove]
        old_data += examples_to_add

        self._data = old_data
        self._operations = self.operations[:-1]
        for e in examples_to_remove:
            del self._example_store._map[e]  # type: ignore

    def search(self, search_query: str, case_sensitive: bool = True) -> List[Example]:
        """Naive search method to quickly identify examples matching the provided substring

        Args:
            search_query (str): Substring to search each example for
            case_sensitive (bool, optional): Consider case of search query and example text

        Returns:
            List[Example]: Matched examples
        """
        search_query = search_query if case_sensitive else search_query.lower()
        out_examples = []

        for example in self.data:
            example_text = example.text if case_sensitive else example.text.lower()
            if search_query in example_text:
                out_examples.append(example)

        return out_examples

    def set_example_store(self, example_store: ExampleStore) -> None:
        """Overwrite the the internal ExampleStore. You probably don't want to call this.
        Used by the Corpus to ensure the ExampleStore of each dataset is complete.

        Args:
            example_store (ExampleStore): ExampleStore to overwrite with
        """
        self._example_store = example_store

    def from_disk(self, path: Union[str, Path]) -> "Dataset":
        """Load Dataset from disk given a path and a loader function that reads the data
        and returns an iterator of Examples

        Args:
            path (Path): path to load from
            loader_func (Callable, optional): Callable that reads a file and returns a List of examples.
                Defaults to [read_jsonl][recon.loaders.read_jsonl]
        """
        path = ensure_path(path)
        state = None
        if (path / ".recon" / self.name).exists():
            state = cast(
                Dict[str, Any], srsly.read_json(path / ".recon" / self.name / "state.json")
            )
            state = DatasetOperationsState(**state)
            self._operations = state.operations

            example_store_path = path / ".recon" / self.name / "example_store.jsonl"
            if example_store_path.exists():
                self._example_store.from_disk(example_store_path)

        data = read_jsonl(path / f"{self.name}.jsonl")
        self._data = data

        for example in self._data:
            self._example_store.add(example)

        if state and self.commit_hash != state.commit:
            # Dataset changed, examples added
            self._operations.append(
                OperationState(
                    name="recon.examples_added_external.v1",
                    status=OperationStatus.COMPLETED,
                    ts=datetime.now(),
                    examples_added=max(len(self) - state.size, 0),
                    examples_removed=max(state.size - len(self), 0),
                    examples_changed=0,
                    transformations=[],
                )
            )

            for op in self._operations:
                op._status = OperationStatus.NOT_STARTED

        operations_to_run: Dict[str, OperationState] = {}
        for op in self._operations:
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

    def to_disk(
        self, output_dir: Union[str, Path], overwrite: bool = False, save_examples: bool = True
    ) -> None:
        """Save Corpus to Disk

        Args:
            output_dir (Path): Output file path to save data to
            overwrite (bool): Force save to directory. Create parent directories
                or overwrite existing data.
            save_examples (bool): Save the example store along with the state.
        """
        output_dir = ensure_path(output_dir)
        state_dir = output_dir / ".recon" / self.name
        if not overwrite and output_dir.exists():
            raise ValueError(
                "Output directory is not empty. Set overwrite=True in Dataset.to_disk to clear the directory before saving."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if not state_dir.exists():
            state_dir.mkdir(parents=True, exist_ok=True)

        state = DatasetOperationsState(
            name=self.name, commit=self.commit_hash, size=len(self), operations=self.operations
        )
        srsly.write_json(state_dir / "state.json", state.dict())

        if save_examples:
            self.example_store.to_disk(state_dir / "example_store.jsonl")

        srsly.write_jsonl(
            output_dir / f"{self.name}.jsonl", [e.dict(exclude_unset=True) for e in self.data]
        )

    def from_prodigy(self, prodigy_datasets: List[str]) -> "Dataset":
        """Need to have from_prodigy accept multiple datasets as a list of str so Prodigy
        can stay separate and new annotation sessions can happen often. Basically prodigy db-merge

        Need to save to only 1 prodigy dataset though for consistency

        Args:
            prodigy_datasets (List[str]): List of prodigy datasets to load from

        Returns:
            Dataset: Initialized dataset with Prodigy data
        """
        from recon.prodigy.utils import from_prodigy

        print(f"Loading data from prodigy datasets: {', '.join(prodigy_datasets)}")
        data = []
        for prodigy_dataset in prodigy_datasets:
            data += from_prodigy(prodigy_dataset)
        self._data = data
        return self

    def to_prodigy(self, prodigy_dataset: Optional[str] = None, overwrite: bool = True) -> str:
        """Save examples to prodigy dataset

        Args:
            prodigy_dataset (str, optional): Prodigy dataset name to save to.
            overwrite (bool, optional): Overwrite dataset name if it exists.

        Returns:
            str: Prodigy dataset name
        """
        from recon.prodigy.utils import to_prodigy

        if not prodigy_dataset:
            prodigy_dataset = f"{self.name}_{self.commit_hash}"

        print(f"Saving dataset to prodigy dataset: {prodigy_dataset}")
        to_prodigy(self.data, prodigy_dataset, overwrite_dataset=overwrite)
        return prodigy_dataset

    def from_spacy(self, path: Path) -> "Dataset":
        """Load Dataset from a file in the .spacy format

        Args:
            path (Path): path to load from

        Returns:
            Dataset: Initialized dataset with Prodigy data
        """
        data = from_spacy(path)
        self._data = list(data)
        return self

    def to_spacy(self, output_dir: Path) -> None:
        """Save data to .spacy file

        Saves file as {output_dir}/{self.name}.spacy

        Args:
            output_dir (Path): Output file path to save data to
        """
        output_dir = ensure_path(output_dir)
        to_spacy(output_dir / (self.name + ".spacy"), self.data)

    def from_hf_dataset(
        self,
        hf_dataset: "HFDataset",
        tokens_prop: str = "tokens",
        labels_prop: str = "ner_tags",
        labels: List[str] = [],
        lang: str = "en",
    ) -> "Dataset":
        nlp = spacy.blank(lang)
        examples = []
        for e in hf_dataset:
            e = cast(Dict[str, Any], e)
            if labels:
                tags = [labels[tag_n] for tag_n in e[labels_prop]]
            else:
                tags = e[labels_prop]
            tokens = e[tokens_prop]
            doc = Doc(nlp.vocab, words=tokens, spaces=[True] * len(tokens), ents=tags)
            spans = [
                Span(text=ent.text, start=ent.start_char, end=ent.end_char, label=ent.label_)
                for ent in doc.ents
            ]
            tokens = [Token(text=t.text, start=t.idx, end=t.idx + len(t), id=t.i) for t in doc]
            examples.append(Example(text=doc.text, spans=spans, tokens=tokens))
        self._data = examples
        return self
