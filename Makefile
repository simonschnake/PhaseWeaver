.PHONY: test smoke

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
MPLCONFIGDIR ?= .tmp-mpl

ifneq ($(wildcard .venv/bin/python),)
PYTHON := .venv/bin/python
PYTEST := $(PYTHON) -m pytest
endif

test:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTEST) tests

smoke:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTEST) tests/core/test_reconstruction.py tests/core/test_measurement.py
