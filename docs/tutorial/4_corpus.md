# Tutorial - Using `Corpus.apply`

In the previous step, we used the `stats.get_ner_stats` function to some stats on our train_data. Now, we want to be able to get these same stats across our train/dev/test split.

SO, ReconNER's `Corpus` class provides a useful method called [`apply`](../../api/corpus)
that takes a `Callable` as a parameter that can run on a list of `Example`s (e.g. `stats.get_ner_stats`)
and run that `Callable` over all the datasets as well as a concatenation of all the datasets so you get the full picture.

!!! tip
    You can pass arbitary `*args` and `**kwargs` to `Corpus.apply` and they will be passed along to the callable you provide as the required argument.

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
    "ents_per_type":{
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
    "ents_per_type":{
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
    "ents_per_type":{
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
    "ents_per_type":{
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

We want our final model to be equally good at extracting these 3 labels of `SKILL`, `PRODUCT` and `JOB_ROLE` so we now know exactly where to invest more time in our annotations effort: getting more examples of `JOB_ROLE`.

!!! note
    This is a VERY small dataset sampled from a much larger NER dataset that's powering part of our work on the new [v3 Text Analytics Cognitive Service](https://azure.kabirkhan.com/en-us/services/cognitive-services/text-analytics/). So here's your glimpse into how we work with data at kabirkhan. Until we fix the lack of annotations for the `JOB_ROLE` label we won't be launching it in production.


## Next Steps

We've only scratched the surface of ReconNER. It's great to have these global stats about our dataset so we can track trends and make sure we're trending in the right direction as we annotate more data. But this data doesn't debug the data we already have. 34 of our 191 `SKILL` annotations in our `train` set might actually be instances where `JOB_ROLE` or `PRODUCT` is more appropriate.

We might have subsets of our data annotated by different people that had a slightly different understanding of the annotation requirements.

In the next step of this tutorial we'll dive into the `insights` module of ReconNER to examine the quality of our existing annotations.
