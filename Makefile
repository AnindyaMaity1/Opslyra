PYTHON ?= python
OUTPUT ?= instance/experiments
SEEDS ?= 0,1,2
FOLDS ?= 3
MAX_ROWS ?= 800

.PHONY: test reproduce

test:
	$(PYTHON) -m pytest -q

reproduce:
	$(PYTHON) experiments.py --output $(OUTPUT) --seeds $(SEEDS) --folds $(FOLDS) --max-rows $(MAX_ROWS) reproduce
