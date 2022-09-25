As far as stats go, we've only seen the basic `get_ner_stats` function. But Recon has a lot of useful statistics for your data. Let's start with Entity Coverage.


## Entity Coverage

To build a good NER model, it's important to have enough representative annotations for the types of predictions you want your model to make. Entity Coverage stats can help you ensure you aren't just annotating the same spans of text for every example in your dataset.

To start working with Entity Coverage, run `get_entity_coverage` on a Dataset. `get_entity_coverage` returns
a list of EntityCoverage data classes which have the annotation label, the span of text, and the count of how many times that annotation text/label combination appears in the Dataset. The final list is sorted by the count, so the most common entities are at the beginning, and the least common are at the end.

Let's take a look at our Train dataset.

```python hl_lines="12"
{!./src/tutorial/6_conll_ec.py!}
```

<div class="termy">

```console
$ python main.py
Most Covered Entities
[EntityCoverage(text='U.S.', label='LOC', count=303),
 EntityCoverage(text='Germany', label='LOC', count=141),
 EntityCoverage(text='Britain', label='LOC', count=133),
 EntityCoverage(text='Australia', label='LOC', count=130),
 EntityCoverage(text='England', label='LOC', count=123),
 EntityCoverage(text='France', label='LOC', count=122),
 EntityCoverage(text='Spain', label='LOC', count=110),
 EntityCoverage(text='Italy', label='LOC', count=98),
 EntityCoverage(text='LONDON', label='LOC', count=93),
 EntityCoverage(text='Russian', label='MISC', count=92)]

Least Covered Entities
[EntityCoverage(text='Luca Cadalora', label='PER', count=1),
 EntityCoverage(text='Alex Criville', label='PER', count=1),
 EntityCoverage(text='Scott Russell', label='PER', count=1),
 EntityCoverage(text='Tadayuki Okada', label='PER', count=1),
 EntityCoverage(text='Carlos Checa', label='PER', count=1),
 EntityCoverage(text='Alexandre Barros', label='PER', count=1),
 EntityCoverage(text='Shinichi Itoh', label='PER', count=1),
 EntityCoverage(text='Swe', label='LOC', count=1),
 EntityCoverage(text='Bob May', label='PER', count=1),
 EntityCoverage(text='Bradley Hughes', label='PER', count=1)]
```

</div>


Looking at the 10 most covered entities and the 10 least covered entities can already help us to understand the Conll 2003 data. Predictably, high profile locations like the "U.S." and other countries are at the top. These countries of course get referenced more often than individual people like "Bob May" or "Bradley Hughes".

But we can also see some notable entries such as "Russian" being labeled as "MISC".
