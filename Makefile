
.DEFAULT_GOAL := all
autoflake = poetry run autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place docs/src/ recon tests --exclude=__init__.py
flake8 = poetry run flake8 --ignore E501 recon tests
isort = poetry run isort recon tests
black = poetry run black -S -l 120 --target-version py39 recon tests


.PHONY: install
install: 
	poetry install


.PHONY: format
format:
	$(autoflake)
	$(isort)
	$(black)

.PHONY: lint
lint:
	# $(flake8)
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: test
test:
	poetry run pytest ./tests --cov=recon --cov-report=term-missing -o console_output_style=progress


