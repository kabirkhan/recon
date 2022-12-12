One of the hardest choices for an NER model to make is when it sees the same text span annotated with 2 different labels in different contexts.

This is also one of the most useful things for a model to learn. For example, lots of people are named after cities they were born in or that have some significance to their parents.

```python
{!./src/tutorial/7_conll_dallas_example.py!}
```

"Dallas" is a person's name in the first example and "Dallas" is a location in the second example (according to CONLL annotation guidelines).

The label is correct in both cases and whichever NER model we want to train will need to rely on the context of the sentence to figure out which label (if any) to assign the span "Dallas" with in future predictions.

That being said, sometimes the distinction between these labels is harder or annotators just make mistakes. To find these sorts of mistakes, we can use Recon's `get_label_disparities` function.

```python
{!./src/tutorial/8_conll_label_disparities.py!}
```
