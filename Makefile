init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest --cov=torchpipe --cov-report=term-missing tests
