
.DEFAULT_GOAL := all
isort = poetry run isort recon tests
black = poetry run black -S -l 120 --target-version py39 recon tests


.PHONY: install
install: 
	poetry install


.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	poetry run flake8 recon/ tests/
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: test
test:
	poetry run pytest ./tests --cov=recon --cov-report=term-missing -o console_output_style=progress


