import functools
from collections import Counter, defaultdict
from copy import deepcopy
from inspect import isclass
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Union

import catalogue
import srsly
from wasabi import Printer

from .preprocess import PreProcessor
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
        with msg.loading(f"\t=> Running preprocessor {processor.name}..."):
            processor_outputs = list(processor(data))
            msg.good("Done")

        for i, (example, output) in enumerate(zip(data, processor_outputs)):
            preprocessed_outputs[example][processor.name] = processor_outputs[i]

    for example in data:
        yield hash(example), example.copy(deep=True), preprocessed_outputs[example]


class operation:
    def __init__(self, name: str, pre: List[PreProcessor] = []):
        """Decorate an operation that makes some changes to a dataset.

        Args:
            name (str): Operation name.
            pre (List[PreProcessor]): List of preprocessors to run
        """
        self.name = name
        self.pre = pre

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
        registry.operations.register(self.name)(Operation(self.name, self.pre, op))

        return op


class Operation:
    """Operation class that takes care of calling and reporting
    the results of an operation on a Dataset"""

    def __init__(self, name: str, pre: List[PreProcessor], op: Callable):
        """Initialize an Operation instance
        
        Args:
            name (str): Name of operation
            pre (List[PreProcessor]): List of preprocessors to run
            op (Callable): Decorated function
        """
        self.name = name
        self.pre = pre
        self.op = op

    def __call__(self, dataset: Any, *args: Any, **kwargs: Any) -> OperationResult:
        """Runs op on a dataset and records the results
        
        Args:
            dataset (Dataset): Dataset to operate on
        
        Raises:
            ValueError: if track_example is called in the op with no data
        
        Returns:
            OperationResult: Container holding new data and the state of the Operation
        """
        verbose = True
        msg = Printer(no_print=verbose == False)

        initial_state = kwargs.pop("initial_state") if "initial_state" in kwargs else None
        verbose = kwargs.pop("verbose") if "verbose" in kwargs else None
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

        new_data = []
        for orig_example_hash, example, preprocessed_outputs in op_iter(
            dataset.data, self.pre, verbose=verbose
        ):
            if preprocessed_outputs:
                res = self.op(example, *args, preprocessed_outputs=preprocessed_outputs, **kwargs)
            else:
                res = self.op(example, *args, **kwargs)

            if res is None:
                remove_example(orig_example_hash)
            elif isinstance(res, list):
                old_example_present = False
                for new_example in res:
                    if hash(new_example) == orig_example_hash:
                        old_example_present = True
                    else:
                        new_data.append(new_example)
                        add_example(new_example)
                if not old_example_present:
                    remove_example(orig_example_hash)
            else:
                assert isinstance(res.text, str)
                assert isinstance(res.spans, list)
                new_data.append(res)
                if hash(res) != orig_example_hash:
                    change_example(orig_example_hash, res)

        transformation_counts = Counter([t.type for t in state.transformations])

        state.examples_added = transformation_counts[TransformationType.EXAMPLE_ADDED]
        state.examples_removed = transformation_counts[TransformationType.EXAMPLE_REMOVED]
        state.examples_changed = transformation_counts[TransformationType.EXAMPLE_CHANGED]
        state.status = OperationStatus.COMPLETED

        state_copy = state.copy(deep=True)
        state = OperationState(name=self.name)
        return OperationResult(data=new_data, state=state_copy)
