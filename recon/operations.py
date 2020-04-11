from collections import Counter
from copy import deepcopy
from inspect import isclass
from typing import Any, Callable, Dict, List, Set, Tuple

import catalogue
import srsly

from .types import Example, OperationResult, OperationState, OperationStatus, Transformation, TransformationCallbacks, TransformationType


class registry:
    operations = catalogue.create("recon", "operations", entry_points=True)


class operation:
    def __init__(self, name: str):
        """Decorate a pipeline component.
        name (str): Default component and factory name.
        """
        self.name = name

    def __call__(self, *args, **kwargs):
        obj = args[0]
        args = args[1:]


        def factory(dataset, *args, **kwargs) -> OperationResult:
            # print(self.initial_state)
            # if self.initial_state.status == OperationStatus.COMPLETED:
            #     return OperationResult(data=dataset.data, state=self.initial_state)
            # TODO: run this step above inside Dataset.apply_
            initial_state = kwargs.pop("initial_state")
            if not initial_state:
                initial_state = OperationState(name=self.name)
            state = initial_state.copy(deep=True)

            print(f"Running operation on {dataset.name}")
            print("=" * 100)

            if state.status == OperationStatus.NOT_STARTED:
                state.status = OperationStatus.IN_PROGRESS

            def add_example(new_example: Example):
                state.transformations.append(
                    Transformation(
                        example=hash(new_example),
                        type=TransformationType.EXAMPLE_ADDED
                    )
                )
                dataset.example_store.add(new_example)


            def remove_example(orig_example_hash: int):
                state.transformations.append(
                    Transformation(
                        prev_example=orig_example_hash,
                        type=TransformationType.EXAMPLE_REMOVED
                    )
                )

            def change_example(orig_example_hash: int, new_example: Example):
                state.transformations.append(
                    Transformation(
                        prev_example=orig_example_hash,
                        example=hash(new_example),
                        type=TransformationType.EXAMPLE_CHANGED
                    )
                )
                dataset.example_store.add(new_example)


            callbacks = TransformationCallbacks(
                add_example=add_example,
                remove_example=remove_example,
                change_example=change_example
            )

            # if isclass(obj):
            #     new_data = obj()(dataset.data, callbacks=callbacks)
            # else:
            new_data = obj(dataset.data, *args, **kwargs, callbacks=callbacks)
            transformation_counts = Counter([t.type for t in state.transformations])

            state.examples_added = transformation_counts[TransformationType.EXAMPLE_ADDED]
            state.examples_removed = transformation_counts[TransformationType.EXAMPLE_REMOVED]
            state.examples_changed = transformation_counts[TransformationType.EXAMPLE_CHANGED]

            print("TRANSFORMATIONS CALCULATED")
            print("Transformations (ADDED): ", state.examples_added)
            print("Transformations (REMOVED): ", state.examples_removed)
            print("Transformations (CHANGED): ", state.examples_changed)
            
            state.status = OperationStatus.COMPLETED

            state_copy = state.copy(deep=True)
            state = OperationState(name=self.name)
            return OperationResult(data=new_data, state=state_copy)

        registry.operations.register(self.name)(factory)
        return obj
