PYTHON ?= python3
PYTHONPATH := src
GUI_HOST ?= 127.0.0.1
GUI_PORT ?= 8501

.PHONY: help install test gui gui-health validate-example demo simulate benchmark-smoke

help:
	@echo "install          Install the package with development dependencies"
	@echo "test             Run the complete test suite"
	@echo "gui              Launch the local research explorer"
	@echo "gui-health       Check a GUI server already running in another terminal"
	@echo "validate-example Validate the included synthetic observation example"
	@echo "demo             Run the legacy circular-model synthetic demonstration"
	@echo "simulate         Simulate a cohort with the reduced first-passage model"
	@echo "benchmark-smoke  Run the short multi-model benchmark"

install:
	$(PYTHON) -m pip install -e ".[dev,data,gui]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

gui:
	@echo "Keep this terminal open while using the GUI."
	@echo "Open http://$(GUI_HOST):$(GUI_PORT) in a browser on this computer."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m streamlit run app/streamlit_app.py --server.address $(GUI_HOST) --server.port $(GUI_PORT) --server.headless true

gui-health:
	@curl --fail --silent --show-error http://$(GUI_HOST):$(GUI_PORT)/_stcore/health

validate-example:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli validate-data data/example_common_observations.csv

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_synthetic_demo.py --out outputs/synthetic_demo

simulate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli simulate --config configs/experiments/milestone_1.yaml --output-dir outputs/simulation

benchmark-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m digital_twin.cli run-benchmark --config configs/benchmarks/smoke.yaml --project-root .
