
.DEFAULT_GOAL := all
black = poetry run black -S -l 88 --target-version py39 --preview recon tests docs/src
pyright = poetry run pyright
ruff = poetry run ruff recon tests docs/src


.PHONY: install
install:
	poetry install


.PHONY: format
format:
	$(black)
	$(ruff) --fix

.PHONY: lint
lint:
	$(ruff)
	$(black) --check --diff

.PHONY: pyright
pyright:
	$(pyright)

.PHONY: test
test:
	poetry run pytest ./tests --cov=recon --cov-report=term-missing -o console_output_style=progress

.PHONY: build-docs
build-docs:
	poetry run python -m mkdocs build
	cp ./docs/index.md ./README.md

.PHONY: deploy-docs
deploy-docs: build-docs
	poetry run python -m mkdocs gh-deploy --force

.PHONY: docs-live
docs-live:
	poetry run mkdocs serve --dev-addr 0.0.0.0:8009
