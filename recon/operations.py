import functools
from collections import Counter
from copy import deepcopy
from inspect import isclass
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Union

import catalogue
import srsly

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


def op_iter(data: List[Example]) -> Iterator[Tuple[int, Example]]:
    """Iterate over list of examples for an operation
    yielding tuples of (example hash, example)
    
    Args:
        data (List[Example]): List of examples to iterate
    
    Yields:
        Iterator[Tuple[int, Example]]: Tuples of (example hash, example)
    """
    for example in data:
        yield hash(example), example.copy(deep=True)


class operation:
    def __init__(self, name: str, batch: bool = False):
        """Decorate an operation that makes some changes to a dataset.

        Args:
            name (str): Operation name.
            batch (bool): Send all examples in dataset for batch operation. 
        """
        self.name = name
        self.batch = batch

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """Decorator for an operation. 
        The first arg is the function being decorated.
        This function can either operate on a List[Example]
        and in that case self.batch should be True.

        e.g. @operation("some_name", batch=True)

        Or it should operate on a single example and 
        recon will take care of applying it to a full Dataset
        
        Args:
            args: First arg is function to decorate
        
        Returns:
            Callable: Original function
        """
        op: Callable = args[0]
        registry.operations.register(self.name)(Operation(self.name, self.batch, op))

        return op


class Operation:
    """Operation class that takes care of calling and reporting
    the results of an operation on a Dataset"""

    def __init__(self, name: str, batch: bool, op: Callable):
        """Initialize an Operation instance
        
        Args:
            name (str): Name of operation
            batch (bool): Whether the operation handles a batch of data or not
            op (Callable): Decorated function
        """
        self.name = name
        self.batch = batch
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
        initial_state = kwargs.pop("initial_state")
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

        def track_example(orig_example_hash: int = None, new_example: Example = None) -> None:
            if orig_example_hash and not new_example:
                remove_example(orig_example_hash)
            elif new_example and not orig_example_hash:
                add_example(new_example)
            elif orig_example_hash and new_example:
                if orig_example_hash != hash(new_example):
                    change_example(orig_example_hash, new_example)
            else:
                raise ValueError("Error tracking example, no data sent")

        callbacks = TransformationCallbacks(
            add_example=add_example,
            remove_example=remove_example,
            change_example=change_example,
            track_example=track_example,
        )

        if self.batch:
            new_data = self.op(dataset.data, *args, **kwargs, callbacks=callbacks)
        else:
            new_data = []
            for orig_example_hash, example in op_iter(dataset.data):
                res: Union[Example, List[Example], None] = self.op(example, *args, **kwargs)
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
                    track_example(orig_example_hash, res)

        transformation_counts = Counter([t.type for t in state.transformations])

        state.examples_added = transformation_counts[TransformationType.EXAMPLE_ADDED]
        state.examples_removed = transformation_counts[TransformationType.EXAMPLE_REMOVED]
        state.examples_changed = transformation_counts[TransformationType.EXAMPLE_CHANGED]
        state.status = OperationStatus.COMPLETED

        state_copy = state.copy(deep=True)
        state = OperationState(name=self.name)
        return OperationResult(data=new_data, state=state_copy)
