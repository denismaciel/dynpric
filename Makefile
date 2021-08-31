install-dev-dependencies:
	pip install pip-tools
	pip-sync requirements-dev.txt
	pip install -Ie .
	pre-commit install
