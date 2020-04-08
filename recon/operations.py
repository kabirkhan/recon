from copy import deepcopy
from inspect import isclass
from typing import Any, Callable, Dict

import catalogue

from .types import OperationResult, OperationState, OperationStatus, Transformation, TransformationType


class registry:
    operations = catalogue.create("recon", "operations", entry_points=True)


class operation:
    def __init__(self, name: str, initial_state: OperationState = None):
        """Decorate a pipeline component.
        name (str): Default component and factory name.
        """
        self.name = name
        
        if not initial_state:
            initial_state = OperationState(name=self.name)
        self.initial_state = initial_state

    def __call__(self, *args, **kwargs):
        obj = args[0]
        args = args[1:]


        def factory(data, **cfg) -> OperationResult:
            state = deepcopy(self.initial_state)
            state.status = OperationStatus.IN_PROGRESS

            if isclass(obj):
                new_data = obj(cfg)(data, state=state)
            else:
                new_data = obj(data, state=state)

            initial_len = len(data)
            new_data_len = len(new_data)

            old_examples = {hash(e) for e in data}
            new_examples = {hash(e) for e in new_data}

            changed_examples = old_examples.difference(new_examples)
            added_examples = new_examples.difference(old_examples)

            state.examples_added = max(new_data_len - initial_len, 0)
            state.examples_removed = max(initial_len - new_data_len, 0)
            state.examples_changed = max(len(changed_examples) - state.examples_added, 0)


            print(changed_examples, added_examples, state.examples_added, state.examples_removed, len(new_data), len(data))
            
            # transformations = []
            # for h in changed_examples:



            state.status = OperationStatus.COMPLETED

            return OperationResult(data=new_data, state=state)

        registry.operations.register(self.name)(factory)
        return obj
