.PHONY: frontend-install frontend-dev frontend-build test lint build verify

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

build: frontend-build
	@echo "Frontend built to src/bot/dashboard/static/"
	@test -f src/bot/dashboard/static/index.html || (echo "ERROR: index.html not found in static/" && exit 1)
	@echo "Verifying Python package..."
	@python -c "import bot; print('Python package OK')"
	@echo "Build complete."

verify: lint test
	@echo "All checks passed."
