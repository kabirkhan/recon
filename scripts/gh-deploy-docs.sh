#!/usr/bin/env bash

poetry run python -m mkdocs gh-deploy

cp ./docs/index.md ./README.md
