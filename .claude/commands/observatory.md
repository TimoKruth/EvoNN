Start the Observatory dashboard. Arguments: $ARGUMENTS

Steps:
1. Kill any existing Observatory process: pkill -9 -f "uvicorn.*8417"
2. Wait 2 seconds for port release
3. cd to /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis
4. Start: uv run python -m symbiosis observatory --evonn-root ../EvoNN --evonn2-root ../EvoNN-2 --port 8417
5. Verify it responds: curl -s http://localhost:8417/api/stats
6. Tell the user to open http://localhost:8417
