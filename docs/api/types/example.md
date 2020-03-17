The following Data Types are used to statically type and manage
dataflow throughout Recon.

## Example

The core data representation for Recon is an Example. It contains all relevant information
for NER annotations for a string of text.

### Attributes

| Name    | Type             | Description                                   | Default    |
|---------|------------------|-----------------------------------------------|------------|
| `text`  | `str`            | Example text                                  | *required* |
| `spans` | `List[Span]`     | List of entity spans                          | *required* |
| `meta`  | `Dict[str, Any]` | Dictionariy of metadata about the example     | `dict()`   |

## Span

Each Example has a List of Span objects in the `spans` property that contain information
about Entities annotated/identified in the Example.

### Attributes

| Name    | Type   | Description                                | Default    |
|---------|--------|--------------------------------------------|------------|
| `text`  | `str`  | Span text                                  | *required* |
| `start` | `int`  | Span start character index in Example text | *required* |
| `end`   | `int`  | Span end character index in Example text   | *required* |
| `label` | `int`  | Entity label                               | *required* |



<!-- text: str
    """Span text"""
    start: int
    """Span start character index in Example text."""
    end: int
    """Span end character index in Example text."""
    label: str
    """Entity label""" -->
<!-- ::: recon.types
 -->