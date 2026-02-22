.PHONY: frontend-install frontend-dev frontend-build test lint

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
