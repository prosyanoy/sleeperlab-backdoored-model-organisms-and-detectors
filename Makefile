PY ?= .venv/bin/python
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	$(PY) -m pytest -q
quick:
	$(PY) -m sleeperlab.run --config configs/quick.json --out results/quick
full:
	$(PY) -m sleeperlab.run --config configs/default.json --out results/default
demo:
	$(PY) scripts/demo.py
