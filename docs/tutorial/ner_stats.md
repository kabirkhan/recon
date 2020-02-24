# Tutorial - NER Statistics

Getting statistics about your NER data is extremely helpful throughout the annotation process. It helps you ensure that you're spendind time on the right annotations and that you have enough examples for each type as well as enough examples with **NO ENTITIES** at all (this is often overlooked but **VERY** important to build a model that generalizes well).

Once you have your data loaded either by itself as a list of `Example`s or as a `Dataset` you can easily get statistics using the [`stats.ner_stats`](../../api/stats#ner_stats) function.

The `stats.ner_stats` function expects a `List[Example]` as it's input parameter and will return a serializable response with info about your data. Let's see how this works on the provided example data.

If you don't already have the example data, download it now:
f
!!! error
    Fix this, add CLI for example data download

## Example

Create a file main.py with:

```Python hl_lines="10"
{!./src/tutorial/1_stats.py!}
```

Run the application with the example data.

<div class="termy">

```console
$ python main.py ./examples/data
{
    "n_examples":102,
    "n_examples_no_entities":29,
    "ents_per_type":{
        "SKILL":191,
        "PRODUCT":34,
        "JOB_ROLE":5
    }
}
```

</div>

But it isn't super helpful to have stats on **just** your training data.
And it'd be really annoying to have to call the same function on each dataset:

```Python
ner_stats(ds.train, serialize=True)
ner_stats(ds.dev, serialize=True)
ner_stats(ds.test, serialize=True)
```

## Next Steps

In the next step step of this tutorial you'll learn about how to remove the above boilerplate and run functions across your train/dev/test Dataset split.
