# Loading your data

ReconNER expects your data to be in the [Prodigy Annotation Format](https://prodi.gy/docs/api-interfaces#ner).

A single example in this format looks like:

```JSON
{
  "text": "Apple updates its analytics service with new metrics",
  "spans": [{ "start": 0, "end": 5, "label": "ORG" }]
}
```

ReconNER expects your data to be in a collection in the `.jsonl` File Format.

## Load Dataset from_disk

There are several utilities available for loading your data.


The easiest way to load your data is to initialize a [Dataset](../api/dataset.md) from disk.
If you have a train/dev/test split or just train/dev files in the same directory, it's as easy as calling the `from_disk` `classmethod` for the `Dataset` object.

```Python
ds = Dataset.from_disk('path/to/data_dir')
```

`Dataset.from_disk` will look in the `data_dir` you provide for a file structure that looks like:

```
data_dir
│   train.jsonl
│   dev.jsonl
│   test.jsonl
```

!!! note
    The test.jsonl file is **optional** but generally you should split your annotated data into train/dev/test files.

## The Process of Loading Data

While it's recommended to load data using the `Dataset.from_disk` method, you can also load data directly from disk using the `loaders.read_jsonl` and `loaders.read_json` functions.

These functions expect the same example format (in fact, the `Dataset.from_disk` runs `loaders.read_jsonl` function) and run a few steps.

### 1. Read data from disk
Loads your data with <a href="https://github.com/explosion/srsly" class="external-link" target="_blank">srsly</a> using `srsly.read_jsonl` or `srsly.read_json`

### 2. Fix Annotation Format
Fixes some common issues in Annotation formatting that can arise using the [`validation.fix_annotations_format`](../../api/validation/#fix_annotations_format)

### 3. Filter Overlapping Entities
Often, you'll find your data has overlapping entities. For instance, imagine you have 2 annotators and one decided "Tesla" is a `PRODUCT` and the other noticed that the sentence is actually about "Tesla Motors" which they label as an `ORG`. This function does it's best to resolve these overlaps and in the case above would select "Tesla Motors" `ORG` as the correct entity, deleting "Tesla" `PRODUCT` from the data [`validation.filter_overlaps`](../../api/validation/#filter_overlaps)

### 4. Load into ReconNER type system

Finally these loaders will take a list of JSON examples in the Prodigy Annotation Format outlined above and convert it into a list of `Example` models using <a href="https://pydantic-docs.helpmanual.io/" class="external-link" target="_blank">Pydantic</a>

## Next Steps

Once you have your data loaded, you can run other ReconNER functions on top of it to gain insights into the quality and completeness of your NER data
