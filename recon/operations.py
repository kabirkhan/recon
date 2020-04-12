from collections import Counter
from copy import deepcopy
import functools
from inspect import isclass
from typing import Any, Callable, Dict, List, Set, Tuple, Union

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
    batch_operations = catalogue.create("recon", "batch_operations", entry_points=True)


def op_iter(data: List[Example], batch: bool = True):
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

    def __call__(self, *args, **kwargs) -> Callable:
        op: Callable = args[0]

        class Operation:
            def __init__(_self):
                _self.name = self.name
                _self.batch = self.batch

            def __call__(_self, dataset, *args, **kwargs) -> OperationResult:
                initial_state = kwargs.pop("initial_state")
                if not initial_state:
                    initial_state = OperationState(name=self.name)
                state = initial_state.copy(deep=True)

                if state.status == OperationStatus.NOT_STARTED:
                    state.status = OperationStatus.IN_PROGRESS

                def add_example(new_example: Example):
                    state.transformations.append(
                        Transformation(example=hash(new_example), type=TransformationType.EXAMPLE_ADDED)
                    )
                    dataset.example_store.add(new_example)

                def remove_example(orig_example_hash: int):
                    state.transformations.append(
                        Transformation(
                            prev_example=orig_example_hash, type=TransformationType.EXAMPLE_REMOVED
                        )
                    )

                def change_example(orig_example_hash: int, new_example: Example):
                    state.transformations.append(
                        Transformation(
                            prev_example=orig_example_hash,
                            example=hash(new_example),
                            type=TransformationType.EXAMPLE_CHANGED,
                        )
                    )
                    dataset.example_store.add(new_example)

                def track_example(orig_example_hash: int = None, new_example: Example = None):
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
                    new_data = op(dataset.data, *args, **kwargs, callbacks=callbacks)
                else:
                    new_data = []
                    for orig_example_hash, example in op_iter(dataset.data):
                        res: Union[Example, List[Example], None] = op(example, *args, **kwargs)
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

        registry.operations.register(self.name)(Operation())

        return op
