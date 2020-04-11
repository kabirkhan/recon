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
    def __init__(self, name: str, initial_state: OperationState = None, inject_state: bool = False):
        """Decorate a pipeline component.
        name (str): Default component and factory name.
        """
        self.name = name
        
        if not initial_state:
            initial_state = OperationState(name=self.name)
        self.initial_state = initial_state
        self.inject_state = inject_state

    def __call__(self, *args, **kwargs):
        obj = args[0]
        args = args[1:]


        def factory(dataset, **cfg) -> OperationResult:
            # print(self.initial_state)
            # if self.initial_state.status == OperationStatus.COMPLETED:
            #     return OperationResult(data=dataset.data, state=self.initial_state)
            # TODO: run this step above inside Dataset.apply_
            state = self.initial_state.copy(deep=True)

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

            if isclass(obj):
                if self.inject_state:
                    new_data = obj(cfg)(dataset.data, state=state)
                else:
                    new_data = obj(cfg)(dataset.data)
            else:
                if self.inject_state:
                    new_data = obj(dataset.data, callbacks=callbacks)
                else:
                    new_data = obj(dataset.data)

            prev_data = dataset.data

            initial_len = len(prev_data)
            new_data_len = len(new_data)

            
            transformation_counts = Counter([t.type for t in state.transformations])
            print("TRANSFORMATIONS CALCULATED")
            print(srsly.json_dumps(transformation_counts, indent=4))
            print("Transformations (ADDED): ", len([t for t in state.transformations if t.type == TransformationType.EXAMPLE_ADDED]))
            print("Transformations (REMOVED): ", len([t for t in state.transformations if t.type == TransformationType.EXAMPLE_REMOVED]))
            print("Transformations (CHANGED): ", len([t for t in state.transformations if t.type == TransformationType.EXAMPLE_CHANGED]))


            state.examples_added = transformation_counts[TransformationType.EXAMPLE_ADDED]# len([t for t in state.transformations if t.type == TransformationType.EXAMPLE_ADDED])
            state.examples_removed = transformation_counts[TransformationType.EXAMPLE_REMOVED]
            state.examples_changed = transformation_counts[TransformationType.EXAMPLE_CHANGED]
            
            state.status = OperationStatus.COMPLETED

            state_copy = state.copy(deep=True)
            state = OperationState(name=self.name)
            yield OperationResult(data=new_data, state=state_copy)

            # def get_examples_and_texts(data: List[Example]) -> Tuple[Dict[int, int], Dict[str, int]]:
            #     examples: Dict[int, int] = {}
            #     texts: Dict[str, int] = {}

            #     for i, e in enumerate(data):
            #         examples[hash(e)] = i
            #         texts[e.text_hash()] = i

            #     return examples, texts

            # prev_example_hashes, prev_texts = get_examples_and_texts(prev_data)
            # new_example_hashes, new_texts = get_examples_and_texts(new_data)


            # from timeit import default_timer as timer

            # start = timer()
            # prev_diff = set(prev_example_hashes).difference(set(new_example_hashes))
            # next_diff = set(new_example_hashes).difference(set(prev_example_hashes))
            # end = timer()
            # print("Total set diff of hashes: ", round(end - start, 2))


            # from dictdiffer import diff
            # start = timer()
            # hash_diff = list(diff(prev_example_hashes, new_example_hashes))
            # end = timer()

            # print("Total diff of hashes: ", round(end - start, 2))

            # start = timer()
            # example_diff = list(diff(prev_data, new_data))
            # end = timer()

            # print("Total diff of examples: ", round(end - start, 2))

            # prev_diff = set(prev_example_hashes).difference(new_examples_hashses)
            # new_diff = set(new_examples_hashses).difference(prev_example_hashes)

            
            # prev_examples_diff = []
            # new_examples_diff = []

            # for pd_hash in prev_diff:
            #     idx = prev_examples_hashses[pd_hash]
                
            #     prev_examples_diff.append(prev_data[idx])



            # added_examples = new_examples.difference(prev_examples)

        registry.operations.register(self.name)(factory)
        return obj
