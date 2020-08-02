#!/usr/bin/env bash

set -e
set -x

mypy recon --disallow-untyped-defs
black recon tests --check
isort recon tests docs/src --check-only
