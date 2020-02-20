#!/usr/bin/env bash

set -e

# PYTHONPATH=./docs/src
# python ./docs/src/setup.py develop

mkdocs serve --dev-addr 0.0.0.0:8009
