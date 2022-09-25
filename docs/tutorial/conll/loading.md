# Introduction to Conll 2003


The Conll 2003 NER Dataset is a widely used, cited, and benchmarked dataset for Named Entity Recognition. The Dataset has four entity types: persons (`PER`), locations (`LOC`), organizations (`ORG`), and names of miscellaneous entities (`MISC`) that don't belong in the other 3 groups.

For the rest of this tutorial, we'll use Recon to find and correct errors in the original Conll 2003 NER dataset.

The Conll 2003 data is publicly available and we'll be utilizing [HuggingFace Datasets](https://huggingface.co/datasets/conll2003) to download it.


!!!tip
    **TL;DR** If you're looking for a shorter version of this tutorial, check out the [Conll 2003 Jupyter Notebook](https://github.com/kabirkhan/recon/blob/main/examples/3.0_hardest_examples.ipynb) in the project `examples` folder.


## Loading data from HuggingFace Datasets

Recon has a specialized loader for HuggingFace Datasets based on the Tabular format of the Conll 2003 data.
An example row of the raw data looks like this.

| id (string) | tokens (json) | pos_tags (json) | chunk_tags (json) | ner_tags (json) |
|-------------|---------------|-----------------|-------------------|-----------------|
| 0 | [ "EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "." ] | [ 22, 42, 16, 21, 35, 37, 16, 21, 7] | [ 11, 21, 11, 12, 21, 22, 11, 12, 0 ]	| [ 3, 0, 7, 0, 0, 0, 7, 0, 0 ] |


We're primaliry interested in the `tokens` and `ner_tags` columns. The `ner_tags` are integer tags from 0-7

The data is already split into train/dev/test datasets so we'll load the whole HF Dataset into a Recon Corpus to get started.

```python
{!./src/tutorial/5_conll_loading.py!}
```

To run, first make sure you have the `datasets` library installed.

```
pip install datasets
```

If you run the code above you'll get the summary statistics for each data split from Recon

<div class="termy">

```console
$ python main.py
Dataset
Name: train
Stats: {
    "n_examples": 14042,
    "n_examples_no_entities": 2910,
    "n_annotations": 23499,
    "n_annotations_per_type": {
        "LOC": 7140,
        "PER": 6600,
        "ORG": 6321,
        "MISC": 3438
    }
}
Dataset
Name: dev
Stats: {
    "n_examples": 3251,
    "n_examples_no_entities": 646,
    "n_annotations": 5942,
    "n_annotations_per_type": {
        "PER": 1842,
        "LOC": 1837,
        "ORG": 1341,
        "MISC": 922
    }
}
Dataset
Name: test
Stats: {
    "n_examples": 3454,
    "n_examples_no_entities": 698,
    "n_annotations": 5648,
    "n_annotations_per_type": {
        "LOC": 1668,
        "ORG": 1661,
        "PER": 1617,
        "MISC": 702
    }
}
```

</div>


## Next Steps

Now that we have the data loaded, let's see what other stats we can get besides the basic ones provided by `Corpus.summary`
