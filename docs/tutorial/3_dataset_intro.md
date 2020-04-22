In Recon, a [`Dataset`](/api/dataset) has a few responsibilities.

* Store exampels
* Store state of **every** mutation made to it using recon operations
* Provide an easy interface to apply functions and pipelines to the dataset data
* Easily serialize and deserialize from/to disk to track state of data across the duration of an annotation project


## Getting Started with Datasets

The easiest way to get started with a `Dataset` is using the from_disk method.

The following example starts by initializing a Dataset with a name ("train") and loading the train.jsonl data for the skills example dataset

Replace the code in main.py with the following

```Python
from pathlib import Path

import typer
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def main(data_file: Path):
    ds = Dataset("train").from_disk(data_file)
    print(get_ner_stats(data, serialize=True))


if __name__ == "__main__":
    typer.run(main)
```

and run with the same command. You should see the exact same result as you did without
using a Dataset. That's because `Dataset.from_disk` calls `read_jsonl`

```console
$ python main.py ./examples/data/skills/train.jsonl
{
    "n_examples":106,
    "n_examples_no_entities":29,
    "n_annotations":243,
    "n_annotations_per_type":{
        "SKILL":197,
        "PRODUCT":33,
        "JOB_ROLE":10,
        "skill":2,
        "product":1
    },
    "examples_with_type":null
}
```

## Applying functions to Datasets

In the previous example we called the get_ner_stats function on the data from the train `Dataset`.
`Dataset` provides a utility function called `apply`. `Dataset.apply` takes any function that operates on a List of Examples and runs it on the Dataset's internal data.

```Python
from pathlib import Path

import typer
from recon.dataset import Dataset
from recon.stats import get_ner_stats


def main(data_file: Path):
    ds = Dataset("train").from_disk(data_file)
    print(ds.apply(get_ner_stats, serialize=True))


if __name__ == "__main__":
    typer.run(main)
```

This might not be that interesting (it doesn't save you a ton of code) but `Dataset.apply` can accept either a function or a name for a registered Recon operation. All functions are registered in a Recon registry.

All functions packaged with recon have "recon.vN..." as  a prefix.

So the above example can be converted to:

```Python
from pathlib import Path

import typer
from recon.dataset import Dataset


def main(data_file: Path):
    ds = Dataset("train").from_disk(data_file)
    print(ds.apply("recon.v1.get_ner_stats", serialize=True))


if __name__ == "__main__":
    typer.run(main)
```

This means you don't have to import the get_ner_stats function. For a full list of operations see
the [operations API guide](/api/operations)

All of these examples should return the exact same response. See for yourself:

<div class="termy">

```console
$ python main.py ./examples/data/skills/train.jsonl
{
    "n_examples":106,
    "n_examples_no_entities":29,
    "n_annotations":243,
    "n_annotations_per_type":{
        "SKILL":197,
        "PRODUCT":33,
        "JOB_ROLE":10,
        "skill":2,
        "product":1
    },
    "examples_with_type":null
}
```
</div>

## Next Steps

It's great that we can manage our data operations using a Dataset and named functions but our data is still messy. We still have those pesky lowercased labels for "skill" and "product" that should clearly be "SKILL" and "PRODUCT" respectively.
In the next step of the tutorial we'll learn how to run operations that mutate a `Dataset` and everything Recon does to keep track of these operations for you.
