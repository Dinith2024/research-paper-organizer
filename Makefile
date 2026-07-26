.PHONY: install run test lint docker

install:
	python -m pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

docker:
	docker compose up --build
