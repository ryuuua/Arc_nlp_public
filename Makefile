PYTHON ?= python3
RESEARCH_SITE_OUTPUT ?= artifacts/delivery/research_site
RESEARCH_SITE_PORT ?= 8765

.PHONY: research-site research-site-check research-site-serve research-site-test

research-site:
	$(PYTHON) scripts/docs/build_site.py build --output "$(RESEARCH_SITE_OUTPUT)"

research-site-check:
	$(PYTHON) scripts/docs/build_site.py check --output "$(RESEARCH_SITE_OUTPUT)"

research-site-serve: research-site-check
	$(PYTHON) -m http.server "$(RESEARCH_SITE_PORT)" --bind 127.0.0.1 --directory "$(RESEARCH_SITE_OUTPUT)"

research-site-test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q -p no:cacheprovider tests/test_research_site.py
