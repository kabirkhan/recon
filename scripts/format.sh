#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place docs/src/ recon tests --exclude=__init__.py
black recon tests docs/src
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --combine-as --line-width 88 --recursive --thirdparty recon --apply recon tests docs/src
