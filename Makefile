PYTHON ?= python3
PYTHONPATH := src

.PHONY: help install test gui validate-example demo simulate benchmark-smoke

help:
	@echo "install          Install the package with development dependencies"
	@echo "test             Run the complete test suite"
	@echo "gui              Launch the local research explorer"
	@echo "validate-example Validate the included synthetic observation example"
	@echo "demo             Run the legacy circular-model synthetic demonstration"
	@echo "simulate         Simulate a cohort with the reduced first-passage model"
	@echo "benchmark-smoke  Run the short multi-model benchmark"

install:
	$(PYTHON) -m pip install -e ".[dev,data,gui]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

gui:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run app/streamlit_app.py

validate-example:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli validate-data data/example_common_observations.csv

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_synthetic_demo.py --out outputs/synthetic_demo

simulate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli simulate --config configs/experiments/milestone_1.yaml --output-dir outputs/simulation

benchmark-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli run-benchmark --config configs/benchmarks/smoke.yaml --project-root .
