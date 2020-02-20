#!/usr/bin/env bash

python -m mkdocs gh-deploy

cp ./docs/index.md ./README.md