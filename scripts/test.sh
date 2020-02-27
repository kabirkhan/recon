#!/usr/bin/env bash

set -e
set -x

# Use xdist-pytest --forked to ensure modified sys.path to import relative modules in examples keeps working
pytest --cov=recon --cov-report=term-missing -o console_output_style=progress ${@}
bash ./scripts/lint.sh
# Include tests for files
bash ./scripts/test-files.sh
