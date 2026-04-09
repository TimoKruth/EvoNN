Show the current status of all three projects. Arguments: $ARGUMENTS

Steps:
1. cd to /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis
2. Run: uv run python -m symbiosis info
3. Check for running processes: ps aux | grep -E "evonn.*evolve|symbiosis.*hybrid|symbiosis.*campaign" | grep -v grep
4. Check Observatory status: curl -s http://localhost:8417/api/stats 2>/dev/null || echo "Observatory not running"
5. Show latest git commit in each project
6. Report any stale WAL files: find /Users/timokruth/Projekte/Evo\ Neural\ Nets/ -name "*.wal" -not -path "*/.venv/*"
