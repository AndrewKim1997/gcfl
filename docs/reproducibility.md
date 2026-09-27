# Reproducibility and provenance

## Original source

The author identifies [`archive/GCFL_main.ipynb`](../archive/GCFL_main.ipynb) as the notebook used directly for the paper experiments. It is an unmodified copy of the file previously stored at `notebooks/original/GCFL_main.ipynb`. Its SHA-256 is `da91b43ceef19ea9bb66e254e94d651be419efa878025361b1a29670ab179d8b`. The extraction tool refuses to regenerate code if this file changes.

The six files in [`experiments/`](../experiments/) contain the joined source lines of the original notebook's experiment code cells, without rewritten methods, changed parameters, or extra wrappers. `01_stylized_simulation.py` comes from cell 1 (zero-based) and covers Tables 1–3. `02_real_world_fl.py` comes from cell 3 for Table 4; `03_estimator_reliability.py` from cell 5 for Table 5; `04_noise_and_auditability.py` from cell 7 for Table 6; `05_high_alignment_metrics.py` from cell 9 for Table 7; and `06_modern_attack.py` from cell 12 for Table 8.

[`src/gcfl/stylized.py`](../src/gcfl/stylized.py) contains the first 284 source lines of cell 1: the configuration, simulator, and price-of-gaming estimator. The package's `__init__.py` selects three names for the example API. The source calculations in `stylized.py` are not rewritten. The example uses smaller illustrative settings and is separate from the paper experiments.

Cell 11 is a Colab shell command, `!pip -q install flwr-datasets[vision] datasets`. It is not valid Python and is not part of a standalone experiment script. Installing the `experiments` extras in `pyproject.toml` supplies those packages locally. Run `python tools/extract_experiments.py --check` to compare every committed extract with its source cell. Run the tool without `--check` to regenerate after a deliberate change to the archived source and its recorded hash.

## Running the experiments

Install the extras with `python -m pip install -e '.[experiments]'` from the repository root and execute one experiment at a time. The stylized simulation uses NumPy and pandas. Tables 4–7 use Fashion-MNIST through torchvision and download it on first use. Table 8 accesses FEMNIST through Flower Datasets and Hugging Face datasets. The extracted scripts keep their original seeds, rounds, settings, and CUDA/CPU selection logic.

The scripts preserve side effects from the notebook. Tables 5, 6, and 7 write `E1_estimator_reliability_results.csv`, `E2_noise_privacy_tradeoff_results.csv`, and `E3_high_alignment_metric_sweep_results.csv` in the current working directory. Table 8 writes a CSV and `.npz` histories under `E4_outputs_noLEAF/`. These are generated artifacts, not input files required by the repository.

The original Table 4 cell prints full-test accuracy alongside head metric, tail welfare, and PoG. Earlier, edited table-formatted notebooks had different presentation behavior. Those earlier files remain accessible in the [commit preceding this reorganization](https://github.com/AndrewKim1997/gcfl/tree/9ee77fefa1fcba734e37bcdb58641d30351336f2/notebooks), rather than as parallel active experiment sources.

## Scope of verification

The attached original notebook was compared with the repository's archived source. All six extracted scripts and the reusable simulator were checked against their source lines and compiled as Python. The small usage example was executed. A CI job checks source equality on future changes. The author has confirmed that the original notebook was used directly for the paper experiments. The federated training experiments and numerical paper results were not rerun for this repository reorganization; exact historical Colab package versions have not been established here.
