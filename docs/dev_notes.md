## Dev notes

- Tests are currently out of date with the dataset/model API (integration paths and slow markers); refresh `tests/` before wiring CI.  
- Makefile/slurm env names need consolidation on `vla`; submit_tests.sh and install target still reference `drawings`.  
- Default paths in `configs/base_config.yml` and `dataset-download.py` point to lab storage; prefer env overrides/relative paths for portability.  
- Preprocess/train/eval SLURM scripts assume GPU; add local CPU-friendly targets when iterating without a cluster.  
- Random baselines are in `results/**/random`; README highlights only line-guided “smart” runs. 
