Run an EvoNN-2 evolution job. Arguments: $ARGUMENTS

Steps:
1. cd to /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-2
2. If a config path is given, run: uv run evonn2 evolve --config <config> --run-dir runs/<run-name>
3. If a benchmark name is given, create a minimal config targeting that benchmark
4. After the run completes, show fitness and topology of the best genome
5. Ask if the user wants to export for Symbiosis comparison
