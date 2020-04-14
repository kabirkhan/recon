## 0.3.2

* Allow Dataset.apply_ to accept a str or operation. If a str is passed, recon attempts to resolve it
in the operations registry.
* Throw an error if plain function passed to Dataset.apply_

## 0.3.1

* Prodigy bug resolution by removing bad entry point

## 0.3.0

* Created Data Lineage System using recon.dataset.Dataset. A `Dataset` is responsible for handling state of its transformations over time
* Created Operations registry. An `operation` is a function that changes a dataset.
