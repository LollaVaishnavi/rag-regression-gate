.PHONY: lint test receipts verify evidence clean

PY ?= python
COV_FLOOR ?= 95

lint:
	ruff check src tests experiments tools

test:
	$(PY) -m pytest \
		--junitxml=reports/junit.xml \
		--cov=ragscore \
		--cov-report=json:reports/coverage.json \
		--cov-report=term-missing \
		-q
	@$(PY) -c "import json, sys; p=json.load(open('reports/coverage.json'))['totals']['percent_covered']; \
print(f'coverage {p:.1f}%'); sys.exit(0 if p >= $(COV_FLOOR) else 1)"

receipts:
	$(PY) tools/collect_metrics.py --skip-tests
	$(PY) tools/check_numbers.py

verify: lint test receipts

evidence:
	$(PY) tools/render_diagrams.py
	$(PY) tools/capture_screenshots.py
	$(PY) tools/record_demo.py

clean:
	rm -rf reports .pytest_cache .ruff_cache **/__pycache__ *.egg-info
