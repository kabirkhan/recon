from pathlib import Path
from typing import List

import srsly

from .types import Example
from .validation import json_to_examples


def read_jsonl(path: Path) -> List[Example]:
    """Read a jsonl annotations file 
    
    ### Parameters
    --------------
    **path**: (Path), required.
        Path to data
    
    ### Returns
    -----------
    (List[Example]): 
        List of Examples
    """

    data = list(srsly.read_jsonl(path))
    return json_to_examples(data)


def read_json(path: Path) -> List[Example]:
    """Read a json annotations file 
    
    ### Parameters
    --------------
    **path**: (Path), required.
        Path to data
    
    ### Returns
    -----------
    (List[Example]): 
        List of Examples
    """
    data = srsly.read_json(path)
    return json_to_examples(data)
