#!/bin/sh -e
set -x

# Sort imports one per line, so autoflake can remove unused imports
isort --recursive  --force-single-line-imports --thirdparty recon --apply recon tests docs/src
sh ./scripts/format.sh
