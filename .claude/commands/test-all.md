Run all tests across all three projects. Arguments: $ARGUMENTS

Steps:
1. Run EvoNN tests: cd /Users/timokruth/Projekte/Evo Neural Nets/EvoNN && uv run python -m pytest -q
2. Run EvoNN-2 tests: cd /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-2 && uv run python -m pytest --deselect tests/test_moe.py::test_moe_gradient_flows -q
3. Run Symbiosis tests: cd /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis && uv run pytest -q
4. Report totals: X EvoNN + Y EvoNN-2 + Z Symbiosis = Total tests
5. Flag any failures
