init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest --cov=pytorch_pipeline --cov-report=term-missing tests
