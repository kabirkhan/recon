#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place docs/src/ recon tests --exclude=__init__.py
black recon tests docs/src
isort recon tests docs/src