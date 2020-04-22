# Tutorial - NER Statistics

Getting statistics about your NER data is extremely helpful throughout the annotation process. It helps you ensure that you're spendind time on the right annotations and that you have enough examples for each type as well as enough examples with **NO ENTITIES** at all (this is often overlooked but **VERY** important to build a model that generalizes well).

Once you have your data loaded either by itself as a list of `Example`s or as a `Dataset` you can easily get statistics using the [`stats.ner_stats`](../../api/stats#ner_stats) function.

The `stats.get_ner_stats` function expects a `List[Example]` as it's input parameter and will return a serializable response with info about your data. Let's see how this works on the provided example data.

<!-- !!! tip
    If you don't already have the example data or a dataset of your own, you can download it now. Open a terminal and run the `download` command.
    ```console
    $ recon download examples ./data
    ``` -->


## Example

Create a file main.py with:

```Python hl_lines="10"
{!./src/tutorial/1_stats.py!}
```

Run the application with the example data and you should see the following results.

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

Great! We have some basic stats about our data but we can already see some issues. Looks like some of our examples are annotated with lowercase labels. These are obviously mistakes and we'll see how to fix these shortly.

But first, it isn't super helpful to have stats on **just** your `train` data.
And it'd be really annoying to have to call the same function on each list of examples:

```Python
train = read_jsonl(train_file)
print(get_ner_stats(train, serialize=True))

dev = read_jsonl(dev_file)
print(get_ner_stats(dev, serialize=True))

test = read_jsonl(test_file)
print(get_ner_stats(test, serialize=True))
```

## Next Steps

In the next step step of this tutorial we'll introduce the core containers Recon uses for managing examples and state:

1. [`Dataset`](/api/dataset) - A `Dataset` has a name and holds a list of examples. Its also responsible for tracking any mutations done to its internal data throught Recon operations. (More on this [later](link_to_operations))

and

2. [`Corpus`](/api/corpus). A `Corpus` is a wrapper around a set of datasets that represent a typical train/eval or train/dev/test split. Using a `Corpus` allows you to gain insights on how well your train set represents your dev/test sets.