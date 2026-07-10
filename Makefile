.PHONY: browser-login-chatgpt browser-test-chatgpt browser-run-chatgpt browser-run

BROWSER_PROFILE_DIR ?= browser_benchmark/browser_profiles
BROWSER_CHANNEL ?= chrome

browser-login-chatgpt:
	python3 browser_benchmark/run_browser_benchmark.py --models chatgpt --channel $(BROWSER_CHANNEL) --profile-dir $(BROWSER_PROFILE_DIR) --login-only

browser-test-chatgpt:
	python3 browser_benchmark/run_browser_benchmark.py --models chatgpt --channel $(BROWSER_CHANNEL) --profile-dir $(BROWSER_PROFILE_DIR) --limit 3

browser-run-chatgpt:
	python3 browser_benchmark/run_browser_benchmark.py --models chatgpt --channel $(BROWSER_CHANNEL) --profile-dir $(BROWSER_PROFILE_DIR)

browser-run:
	python3 browser_benchmark/run_browser_benchmark.py --channel $(BROWSER_CHANNEL) --profile-dir $(BROWSER_PROFILE_DIR)
