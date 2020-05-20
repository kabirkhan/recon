Now that we have our data managed in a Recon `Dataset`, we can make corrections to our data automatically and Recon will take care of keeping track of all operations and transformations run on our data.

The key is the `Dataset.apply_` funciton. 

!!!tip
    It's a common python convention that as far as I know was popularized by PyTorch to have a function return a value (i.e. `apply`) and a that same function name followed by an underscore (i.e. `apply_`) operate on that data inplace.

## Correcting a Dataset

`Dataset.apply_` requires a registered in-place operation that will run across all examples in the Dataset's data.

Let's see an example.

```Python hl_lines="15"
{!./src/tutorial/3_dataset_mutate.py!}
```

<div class="termy">

```console
$ python main.py examples/data/skills/train.jsonl

STATS BEFORE
============
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
STATS AFTER
===========
{
    "n_examples":106,
    "n_examples_no_entities":29,
    "n_annotations":243,
    "n_annotations_per_type":{
        "SKILL":199,
        "PRODUCT":34,
        "JOB_ROLE":10
    },
    "examples_with_type":null
}
```
</div>

Nice! We've easily applied the built-in "upcase_labels" function from Recon to fix our obvious mistakes.

But that's not all...

## Tracking operations

It would be really easy to lose track of the operations run on our data if we ran a bunch of operations. Even with our single operation, we'd have to save a copy of the data before running the `upcase_labels` operation and drill into both versions of the dataset to identify which examples we actually changed. Recon takes care of this tracking for us.

Let's extend our previous example by saving our new Dataset to disk using (conveniently) `Dataset.to_disk`. 


```Python hl_lines="21"
{!./src/tutorial/3_dataset_mutate_save.py!}
```

<div class="termy">

```console
$ python main.py examples/data/skills/train.jsonl examples/fixed_data/skills/train.jsonl

STATS BEFORE
============
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
STATS AFTER
===========
{
    "n_examples":106,
    "n_examples_no_entities":29,
    "n_annotations":243,
    "n_annotations_per_type":{
        "SKILL":199,
        "PRODUCT":34,
        "JOB_ROLE":10
    },
    "examples_with_type":null
}

```
</div>

This should have the same console output as before.

Let's investigate what Recon saved.


```Python hl_lines="15"
{!./src/tutorial/3_dataset_mutate.py!}
```

<div class="termy">

```console
$ ll -a examples/fixed_data/skills

├── .recon
├── train.jsonl

// train.jsonl is just our serialized data from our train Dataset
// What's in .recon?

$ tree -a examples/fixed_data/skills/

examples/fixed_data/skills/
├── .recon
│   ├── example_store.jsonl
│   └── train
│       └── state.json
└── train.jsonl

// Let's investigate the state.json for our train dataset.

$ cat examples/fixed_data/skills/.recon/train/state.json

{
    "name": "train",
    "commit_hash": "375a4cdec36fa9e0efd1d7b2a02fb4287b99dbdc",
    "operations": [
        {
            "name":"recon.v1.upcase_labels",
            "args":[],
            "kwargs":{},
            "status":"COMPLETED",
            "ts":1586687281,
            "examples_added":0,
            "examples_removed":0,
            "examples_changed":3,
            "transformations":[
                {
                    "prev_example":1923088532738022750,
                    "example":1401028415299739275,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":459906967662468309,
                    "example":1525998324968157929,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":200276835658424828,
                    "example":407710308633891847,
                    "type":"EXAMPLE_CHANGED"
                }
            ]
        }
    ]
}
```
</div>

In the last command above you can see the output of what Recon saves when you call `Dataset.to_disk`.

Let's dig into the saved state a bit more.

## Dataset state

The first property stored is the dataset name. Pretty self-explanatory.
The second, `commit`, is a bit more complex. 

!!!tip
    A core principal of Recon is that all the data types can be hashed deterministically. This means you'll get the same hash if across Python environments and sessions for each core data type including: Corpus, Dataset, Example, Span and Token. 

The `commit` property is a SHA-1 hash of the dataset
name combined with that hash of each example in the dataset.
If you're familiar with how [git](https://git-scm.com/) works the idea is pretty similar.

The `commit` property of a dataset allows us to understand if a Dataset changes between operations. 
This can happen if you add new examples and want to rerun or run new operations later based on insights from that new data.

```json hl_lines="4 6 7 8 9 10"
{
    "name": "train",
    "commit_hash": "375a4cdec36fa9e0efd1d7b2a02fb4287b99dbdc",
    "operations": [
        {
            "name":"recon.v1.upcase_labels",
            "args":[],
            "kwargs":{},
            "status":"COMPLETED",
            "ts":1586687281,
            "examples_added":0,
            "examples_removed":0,
            "examples_changed":3,
            "transformations":[
                {
                    "prev_example":1923088532738022750,
                    "example":1401028415299739275,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":459906967662468309,
                    "example":1525998324968157929,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":200276835658424828,
                    "example":407710308633891847,
                    "type":"EXAMPLE_CHANGED"
                }
            ]
        }
    ]
}
```

The core of the stored state is the `operations` property. This operations property has all the information needed to both track and re-run an operation on a dataset.

In the above state we have 1 operation since that's all we've run on our dataset so far.

Each operation has a `name` (in this case `"recon.v1.upcase_labels"`) as well as any python `args` or `kwargs` run with the function. The `upcase_labels` operation has no required parameters so these are empty (we'll see some examples where these are not empty later in the tutorial).

We also have a `status` (one of: NOT_STARTED, IN_PROGRESS, COMPLETED) and a `ts` (timestamp of when the operation was run).

These attributes provide the base information to re-create the exact function call and provide the base information of the operation.

The rest of the properties deal with transformation tracking.
The `examples_added`, `examples_removed`, `examples_changed` give you a summary of the overall changes by the operation.

```json hl_lines="11 12 13"
{
    "name": "train",
    "commit_hash": "375a4cdec36fa9e0efd1d7b2a02fb4287b99dbdc",
    "operations": [
        {
            "name":"recon.v1.upcase_labels",
            "args":[],
            "kwargs":{},
            "status":"COMPLETED",
            "ts":1586687281,
            "examples_added":0,
            "examples_removed":0,
            "examples_changed":3,
            "transformations":[
                {
                    "prev_example":1923088532738022750,
                    "example":1401028415299739275,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":459906967662468309,
                    "example":1525998324968157929,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":200276835658424828,
                    "example":407710308633891847,
                    "type":"EXAMPLE_CHANGED"
                }
            ]
        }
    ]
}
```


Finally, the `transformations` property is the most useful for actually auditing and tracking your data changes.
Each transformation has a `prev_example`, `example` and transformation `type`. 

The example properties contain the Example hash or the example before and after the transformation occured. This is really not useful by itself, but these hashes coincide to the hash -> Example mappings in the example_store.jsonl file that Recon saves for you. The ExampleStore is a central store that keeps track of all examples you've ever had in your dataset. This way, we can always revert back or see a concrete comparison of what each operation added/removed/changed by resolving the transformations to their corresponding examples.

!!!note
    Having an ExampleStore is obviously more than doubling the storage required for your data but NER datasets are ususually not that big since they're hard to annotate. For reference, Recon has been tested on a Dataset of 200K examples with no issue.

The transformation `type` will always be one of (EXAMPLE_ADDED, EXAMPLE_REMOVED, or EXAMPLE_CHANGED).

* EXAMPLE_ADDED - In this case, the `prev_example` property will be `null` since we're just adding an example to our dataset.
    This can happen if an operation returns more than one example for every example it sees. A good reference example is the [`recon.v1.split_sentences`](link/to/split_sentences) operation. This operation will find all the sentences in an example and split them out into separate examples. 

!!!tip
    So if an example has 2 sentences, Recon will track this operation as removing the original example and adding 2 examples. You'll see those reflected in the transformations

* EXAMPLE_REMOVED - By default, Recon removes Examples that have bad final annotations that can't be properly resolved to token boundaries. Good reference examples for this behavior are the operations [`recon.v1.fix_tokenization_and_spacing`](link/to/fix_tokenization_and_spacing) and [`recon.v1.add_tokens`](link/to/add_tokens)



```json hl_lines="14 16 17 18"
{
    "name": "train",
    "commit_hash": "375a4cdec36fa9e0efd1d7b2a02fb4287b99dbdc",
    "operations": [
        {
            "name":"recon.v1.upcase_labels",
            "args":[],
            "kwargs":{},
            "status":"COMPLETED",
            "ts":1586687281,
            "examples_added":0,
            "examples_removed":0,
            "examples_changed":3,
            "transformations":[
                {
                    "prev_example":1923088532738022750,
                    "example":1401028415299739275,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":459906967662468309,
                    "example":1525998324968157929,
                    "type":"EXAMPLE_CHANGED"
                },
                {
                    "prev_example":200276835658424828,
                    "example":407710308633891847,
                    "type":"EXAMPLE_CHANGED"
                }
            ]
        }
    ]
}
```
