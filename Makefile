
.DEFAULT_GOAL := all
isort = isort pydantic tests
black = black -S -l 120 --target-version py38 pydantic tests


.PHONY: install
install: 
	poetry install


.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	flake8 pydantic/ tests/
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: test
test:
	pytest ./tests --cov=recon --cov-report=term-missing -o console_output_style=progress


