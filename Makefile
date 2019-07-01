init:
	pipenv install --skip-lock --dev
	pipenv run pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
test:
	pipenv run pytest --cov=torchpipe --cov-report=term-missing tests
