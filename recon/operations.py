import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import catalogue
from tqdm import tqdm
from wasabi import Printer

from .preprocess import PreProcessor
from .preprocess import registry as pre_registry
from .types import (
    Example,
    OperationResult,
    OperationState,
    OperationStatus,
    Transformation,
    TransformationCallbacks,
    TransformationType,
)


class registry:
    operations = catalogue.create("recon", "operations", entry_points=True)
    operation_factories = catalogue.create("recon", "operation_factories", entry_points=True)


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
    msg = Printer(no_print=verbose == False, hide_animation=verbose == False)
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

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """Decorator for an operation.
        The first arg is the function being decorated.
        This function can either operate on a List[Example]
        and in that case self.batch should be True.

        e.g. @operation("recon.v1.some_name", batch=True)

        Or it should operate on a single example and
        recon will take care of applying it to a full Dataset

        Args:
            args: First arg is function to decorate

        Returns:
            Callable: Original function
        """
        op: Callable = args[0]

        pre: List[PreProcessor] = []

        for pre_name_or_op in self.pre:
            preprocessor = pre_name_or_op
            if isinstance(preprocessor, str):
                preprocessor = pre_registry.preprocessors.get(pre_name_or_op)
            assert isinstance(preprocessor, PreProcessor)
            pre.append(preprocessor)

        if self.factory:

            def factory(pre: List[PreProcessor]) -> Operation:
                return Operation(
                    self.name,
                    pre,
                    op=op,
                    handles_tokens=self.handles_tokens,
                    augmentation=self.augmentation,
                )

            registry.operation_factories.register(self.name)(factory)
        else:
            registry.operations.register(self.name)(
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
            op (Callable): Decorated function
        """
        self.name = name
        self.pre = pre
        self.op = op
        self.handles_tokens = handles_tokens
        self.augmentation = augmentation

    def __call__(self, dataset: Any, *args: Any, **kwargs: Any) -> OperationResult:
        """Runs op on a dataset and records the results

        Args:
            dataset (Dataset): Dataset to operate on

        Raises:
            ValueError: if track_example is called in the op with no data

        Returns:
            OperationResult: Container holding new data and the state of the Operation
        """
        verbose = kwargs.pop("verbose", False)
        initial_state = kwargs.pop("initial_state", None)
        if not initial_state:
            initial_state = OperationState(name=self.name)
        state = initial_state.copy(deep=True)

        if state.status == OperationStatus.NOT_STARTED:
            state.status = OperationStatus.IN_PROGRESS

        def add_example(new_example: Example) -> None:
            state.transformations.append(
                Transformation(example=hash(new_example), type=TransformationType.EXAMPLE_ADDED)
            )
            dataset.example_store.add(new_example)

        def remove_example(orig_example_hash: int) -> None:
            state.transformations.append(
                Transformation(
                    prev_example=orig_example_hash, type=TransformationType.EXAMPLE_REMOVED
                )
            )

        def change_example(orig_example_hash: int, new_example: Example) -> None:
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
                "This dataset seems to have preset tokens. " +
                f"Operation: {self.name} is not currently capable of handling tokens and you will "
                "need to reset tokenization after this operation. " +
                "Applying the `recon.v1.add_tokens` operation after this " +
                "is complete will get you back to a clean state."
                # fmt: on
            )

        new_data = []

        with tqdm(total=len(dataset), disable=(not verbose)) as pbar:
            for orig_example_hash, example, preprocessed_outputs in op_iter(
                dataset.data, self.pre, verbose=verbose
            ):
                if preprocessed_outputs:
                    res = self.op(
                        example, *args, preprocessed_outputs=preprocessed_outputs, **kwargs
                    )
                else:
                    res = self.op(example, *args, **kwargs)

                if res is None:
                    remove_example(orig_example_hash)
                elif isinstance(res, list):
                    old_example_present = False
                    for new_example in res:
                        new_data.append(new_example)
                        if hash(new_example) == orig_example_hash:
                            old_example_present = True
                        else:
                            add_example(new_example)
                    if not old_example_present:
                        remove_example(orig_example_hash)
                else:
                    assert isinstance(res, Example)
                    new_data.append(res)
                    if hash(res) != orig_example_hash:
                        change_example(orig_example_hash, res)

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
        registry.operations.register(self.name)(self)
