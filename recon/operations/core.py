import warnings
from collections import Counter, defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic.error_wrappers import ErrorWrapper
from tqdm import tqdm
from wasabi import Printer

from recon.operations import registry as op_registry
from recon.operations.utils import (
    get_received_operation_data,
    get_required_operation_params,
    request_body_to_args,
)
from recon.preprocess import PreProcessor
from recon.preprocess import registry as pre_registry
from recon.types import (
    Example,
    OperationResult,
    OperationState,
    OperationStatus,
    Transformation,
    TransformationType,
)

if TYPE_CHECKING:
    from recon import Dataset


def op_iter(
    data: List[Example], pre: List[PreProcessor], verbose: bool = True
) -> Iterator[Tuple[int, Example, Dict[str, Any]]]:
    """Iterate over list of examples for an operation
    yielding tuples of (example hash, example)

    Args:
        data (List[Example]): List of examples to iterate
        pre (List[PreProcessor]): List of preprocessors to run
        verbose (bool, optional): Show verbose output.

    Yields:
        Iterator[Tuple[int, Example]]: Tuples of (example hash, example)
    """
    msg = Printer(no_print=not verbose, hide_animation=not verbose)
    preprocessed_outputs: Dict[Example, Dict[str, Any]] = defaultdict(dict)
    for processor in pre:
        msg.info(f"\t=> Running preprocessor {processor.name}")
        processor_outputs = processor(data)
        for example, output in tqdm(
            zip(data, processor_outputs), total=len(data), disable=(not verbose), leave=False
        ):
            preprocessed_outputs[example][processor.name] = output
            example.__setattr__(processor.field, output)

    for example in data:
        yield hash(example), example.copy(deep=True), preprocessed_outputs[example]


class operation:
    def __init__(
        self,
        name: str,
        *,
        pre: List[Union[str, PreProcessor]] = [],
        handles_tokens: bool = True,
        factory: bool = False,
        augmentation: bool = False,
    ):
        """Decorate an operation that makes some changes to a dataset.

        Args:
            name (str): Operation name.
            pre (Union[List[str], List[PreProcessor]]): List of preprocessors to run
        """
        self.name = name
        self.pre = pre
        self.handles_tokens = handles_tokens
        self.factory = factory
        self.augmentation = augmentation

    def __call__(
        self, op: Callable[..., Union[Example, Iterable[Example], None]]
    ) -> Callable[..., Union[Example, Iterable[Example], None]]:
        """Decorator for an operation.
        The first arg to the op callable needs to be an Example.

        e.g. @operation("recon.some_name", batch=True)

        Or it should operate on a single example and
        recon will take care of applying it to a full Dataset

        Args:
            op (Op): First arg is callable to decorate

        Returns:
            Op: Original operation callable
        """
        pre: List[PreProcessor] = []

        for preprocessor in self.pre:
            if isinstance(preprocessor, str):
                preprocessor = pre_registry.preprocessors.get(preprocessor)
            assert isinstance(preprocessor, PreProcessor)
            pre.append(preprocessor)

        op_registry.operations.register(self.name)(
            Operation(self.name, pre, op, self.handles_tokens, augmentation=self.augmentation)
        )
        return op


class Operation:
    """Operation class that takes care of calling and reporting
    the results of an operation on a Dataset"""

    def __init__(
        self,
        name: str,
        pre: List[PreProcessor],
        op: Callable,
        handles_tokens: bool,
        augmentation: bool,
    ):
        """Initialize an Operation instance

        Args:
            name (str): Name of operation
            pre (List[PreProcessor]): List of preprocessors to run
            op (Op): Operation callable
        """
        self.name = name
        self.pre = pre
        self.op = op
        self.handles_tokens = handles_tokens
        self.augmentation = augmentation

    def __call__(
        self,
        dataset: "Dataset",
        *args: Any,
        verbose: bool = False,
        initial_state: Optional[OperationState] = None,
        **kwargs: Any,
    ) -> OperationResult:
        """Runs op on a dataset and records the results

        Args:
            dataset (Dataset): Dataset to operate on

        Raises:
            ValueError: if track_example is called in the op with no data

        Returns:
            OperationResult: Container holding new data and the state of the Operation
        """
        if not initial_state:
            initial_state = OperationState(name=self.name)
        state = initial_state.copy(deep=True)

        if state.status == OperationStatus.NOT_STARTED:
            state.status = OperationStatus.IN_PROGRESS

        state.args = args
        state.kwargs = kwargs

        def track_add_example(new_example: Example) -> None:
            state.transformations.append(
                Transformation(example=hash(new_example), type=TransformationType.EXAMPLE_ADDED)
            )
            dataset.example_store.add(new_example)

        def track_remove_example(orig_example_hash: int) -> None:
            state.transformations.append(
                Transformation(
                    prev_example=orig_example_hash, type=TransformationType.EXAMPLE_REMOVED
                )
            )

        def track_change_example(orig_example_hash: int, new_example: Example) -> None:
            state.transformations.append(
                Transformation(
                    prev_example=orig_example_hash,
                    example=hash(new_example),
                    type=TransformationType.EXAMPLE_CHANGED,
                )
            )
            dataset.example_store.add(new_example)

        has_tokens = False
        for e in dataset.data:
            if e.tokens or any([(s.token_start or s.token_end) for s in e.spans]):
                has_tokens = True
                break

        if has_tokens and not self.handles_tokens:
            warnings.warn(
                # fmt: off
                "This dataset seems to have preset tokens. "
                f"Operation: {self.name} is not currently capable of handling tokens and you will "
                "need to reset tokenization after this operation. "
                "Applying the `recon.add_tokens.v1` operation after this "
                "operation is complete will get you back to a clean state."
                # fmt: on
            )
            state.status = OperationStatus.NEEDS_TOKENIZATION

        required_params = get_required_operation_params(self.op)
        received_data = get_received_operation_data(required_params, state)

        values: Dict[str, Any] = {}
        errors: List[ErrorWrapper] = []

        if received_data:
            values, errors = request_body_to_args(list(required_params.values()), received_data)

        if errors:
            error_msg = (
                f"Validation error while trying to call operation: {self.name} "
                + "with provided args and kwargs values. "
            )
            for err in errors:
                error_msg += str(err.exc)
                print(values)
            raise ValueError(error_msg)

        state.args = ()
        state.kwargs = values

        new_data = []
        with tqdm(total=len(dataset), disable=(not verbose)) as pbar:
            it = op_iter(dataset.data, self.pre, verbose=verbose)
            for orig_example_hash, example, preprocessed_outputs in it:
                if preprocessed_outputs:
                    res = self.op(example, preprocessed_outputs=preprocessed_outputs, **values)
                else:
                    res = self.op(example, **values)

                if res is None:
                    track_remove_example(orig_example_hash)
                elif isinstance(res, list):
                    old_example_present = False
                    for new_example in res:
                        new_data.append(new_example)
                        if hash(new_example) == orig_example_hash:
                            old_example_present = True
                        else:
                            track_add_example(new_example)
                    if not old_example_present:
                        track_remove_example(orig_example_hash)
                else:
                    assert isinstance(res, Example)
                    new_data.append(res)
                    if hash(res) != orig_example_hash:
                        track_change_example(orig_example_hash, res)

                pbar.update(1)

        transformation_counts = Counter([t.type for t in state.transformations])
        state.examples_added = transformation_counts[TransformationType.EXAMPLE_ADDED]
        state.examples_removed = transformation_counts[TransformationType.EXAMPLE_REMOVED]
        state.examples_changed = transformation_counts[TransformationType.EXAMPLE_CHANGED]
        state.status = OperationStatus.COMPLETED

        state_copy = state.copy(deep=True)
        state = OperationState(name=self.name)
        return OperationResult(data=new_data, state=state_copy)

    def register(self) -> None:
        op_registry.operations.register(self.name)(self)
