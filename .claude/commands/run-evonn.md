Run an EvoNN evolution job. Arguments: $ARGUMENTS

Steps:
1. cd to /Users/timokruth/Projekte/Evo Neural Nets/EvoNN
2. If a config path is given, run: uv run evonn evolve run --config <config>
3. If a pack name is given without config, create a minimal config and run it
4. After the run completes, show the run ID and key metrics from state.json
5. Ask if the user wants to export for Symbiosis comparison
