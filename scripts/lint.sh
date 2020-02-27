#!/usr/bin/env bash

set -e
set -x

mypy recon --disallow-untyped-defs
black recon tests --check
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --combine-as --line-width 88 --recursive --check-only --thirdparty recon
