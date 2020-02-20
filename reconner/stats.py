from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional
import srsly

from .constants import NONE
from .types import Example


def ner_stats(data: List[Example],
              serialize: bool = False,
              no_print: bool = False) -> Optional[List[Example]]:
    """Compute statistics for NER data
    
    ### Parameters
    --------------
    **data**: (List[Example]), required.
        Data as a List of Examples
    **serialize**: (bool, optional), Defaults to False.
        Serialize to a JSON string for printing
    **no_print**: (bool, optional), Defaults to False.
        Don't print, return serialized string. Requires serialize to be True
    
    ### Returns
    -----------
    (List[Example]): 
        List of examples or string if serialize and no_print are both True
    """    
    labels: DefaultDict[str, Any] = defaultdict(int)
    examples: DefaultDict[str, Any] = defaultdict(list)
    n_examples_no_entities = 0
    for e in data:
        if not e.spans:
            n_examples_no_entities += 1
            examples[NONE].append(e)
        else:
            for s in e.spans:
                labels[s.label] += 1
                examples[s.label].append(e)

    res = {
        'n_examples': len(data),
        'n_examples_no_entities': n_examples_no_entities,
        'ents_per_type': labels
    }
    if serialize:
        s = srsly.json_dumps(res, indent=4)
        if no_print:
            return s
        else:
            print(s)
    else:
        res['examples_with_type'] = examples
        return res