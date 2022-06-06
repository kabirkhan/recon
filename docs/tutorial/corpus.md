So far, we have been operating on a Dataset which represents a single split of our data. Recon's `Corpus` container allows us to work with our full train/dev/test split by managing a separate `Dataset` for each split. The `Corpus` handles either 2 (train, dev) or 3 (train, dev, test) Datasets. If you happen to split up your data in some other way, you may need to just manage the lower level `Dataset` for each split on your own.

Recon's `Corpus` class provides the same `apply` method as `Dataset` that accepts an `Operation`.
S

## Update script to use `Dataset.apply`

Let's edit that `main.py` file you created in the previous step to utilize the `Corpus.apply` method.

```Python hl_lines="10"
{!./src/tutorial/2_corpus_apply.py!}
```

## Run the application

Now, if you run your script you should get the following output:

<div class="termy">

```console
$ python main.py ./examples/data
train
==================================================
{
    "n_examples":102,
    "n_examples_no_entities":29,
    "n_annotations_per_type":{
        "SKILL":191,
        "PRODUCT":34,
        "JOB_ROLE":5
    }
}
dev
==================================================
{
    "n_examples":110,
    "n_examples_no_entities":49,
    "n_annotations_per_type":{
        "SKILL":159,
        "PRODUCT":20,
        "JOB_ROLE":1
    }
}
test
==================================================
{
    "n_examples":96,
    "n_examples_no_entities":38,
    "n_annotations_per_type":{
        "PRODUCT":35,
        "SKILL":107,
        "JOB_ROLE":2
    }
}
all
==================================================
{
    "n_examples":308,
    "n_examples_no_entities":116,
    "n_annotations_per_type":{
        "SKILL":457,
        "PRODUCT":89,
        "JOB_ROLE":8
    }
}
```

</div>

## Analyzing the results

Now that we have a good understanding of the distribution of labels in across our train/dev/test split as well as the summation of all those numbers to the "all" data, we can start to see some issues.


### 1. Not enough `JOB_ROLE` annotations

 We clearly don't have enough annotations of the `JOB_ROLE` in our data. There's no way an NER model could learn to capture `JOB_ROLE` in a generic way with only 8 total annotations.

### 2. Barely enough `PRODUCT` annotations

We're also a little low (though not nearly as much) on our `PRODUCT` label.

### What to do from here

We want our final model to be equally good at extracting these 3 labels of `SKILL`, `PRODUCT` and `JOB_ROLE` so we now know exactly where to invest more time in our annotations effort: getting more examples of `JOB_ROLE` and `PRODUCT`.

## Next Steps

We've only scratched the surface of Recon. It's great to have these global stats about our dataset so we can make sure we're trending in the right direction as we annotate more data. But this information doesn't help us debug the data we already have.

For example, 34 of our 191 `SKILL` annotations in our `train` set might actually be instances where `JOB_ROLE` or `PRODUCT` is more appropriate.

Or, we might have subsets of our data annotated by different people that had a slightly different understanding of the annotation requirements (or just made a couple mistakes), creating disparities in the final dataset.

In the next step of this tutorial we'll put away the toy skills dataset and take a look at the widely used Conll 2003 Benchmark Dataset. We'll use Recon to find and correct errors in the original dataset and publish our new and improved Conll 2003 dataset.
