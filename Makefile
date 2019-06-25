init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest --cov=torchdata --cov-report=term-missing tests
