.PHONY: install test train clean

install:
	uv sync

test:
	uv run pytest -q

train:
	uv run gridmamba --epochs 20

clean:
	rm -rf outputs/*.pt .pytest_cache
