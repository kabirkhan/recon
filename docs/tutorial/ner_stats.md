# Tutorial - NER Statistics

Getting statistics about your NER data is extremely helpful throughout the annotation process. It helps you ensure that you're spendind time on the right annotations and that you have enough examples for each type as well as enough examples with **NO ENTITIES** at all (this is often overlooked but **VERY** important to build a model that generalizes well).

Once you have your data loaded either by itself as a list of `Example`s or as a `Dataset` you can easily get statistics using the [`stats.ner_stats`](../../api/stats#ner_stats) funtion