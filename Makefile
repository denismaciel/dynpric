install-dev-dependencies:
	pip install pip-tools
	pip-sync requirements.txt
	pip install -Ie .
	pre-commit install
