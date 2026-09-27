# Reproducibility and provenance

## Source of the paper experiments

The author identifies `notebooks/original/GCFL_main.ipynb` as the notebook used directly for the paper experiments. Its SHA-256 is `da91b43ceef19ea9bb66e254e94d651be419efa878025361b1a29670ab179d8b`. The notebook stays at its pre-existing path. The extraction tool refuses to regenerate scripts if this notebook changes.

`experiments/01_stylized_simulation.py` is original code cell 1 (zero-based), covering Tables 1–3. `02_real_world_fl.py` is cell 3 for Table 4; `03_estimator_reliability.py` is cell 5 for Table 5; `04_noise_and_auditability.py` is cell 7 for Table 6; `05_high_alignment_metrics.py` is cell 9 for Table 7; and `06_modern_attack.py` is cell 12 for Table 8. These files contain precisely the joined source lines from those code cells, without rewritten methods, parameter changes, or extra wrappers.

Cell 11 contains a Colab shell command, `!pip -q install flwr-datasets[vision] datasets`. It is not valid Python and is not part of an experiment script. Installing `requirements.txt` supplies those packages for a local run. Run `python tools/extract_experiments.py --check` to verify that every committed script still matches its source cell. Run the tool without `--check` to regenerate the scripts after a deliberate change to the source and its recorded hash.

## Existing table-formatted notebooks

The six `notebooks/Table_*.ipynb` files are separate, edited derivatives of the original notebook's experiments. They are retained at their existing URLs and should not be described as byte-identical extracts. For example, the published Table 4 includes full-test accuracy, which the original cell prints, while the separate table-formatted Table 4 notebook's final data frame omits that column. The new `experiments/` scripts prioritize fidelity to the original experimental record over reproducing the edited notebooks' presentation.

## Running the experiments

Install `requirements.txt` from the repository root and execute one experiment script at a time. The stylized simulation uses NumPy and pandas. Tables 4–7 use Fashion-MNIST through torchvision and download it on first use; Table 8 accesses FEMNIST through Flower Datasets and Hugging Face datasets. The original scripts choose CUDA when available and otherwise use CPU where their source specifies this behavior. The code keeps its original seeds, rounds, and other settings.

The scripts preserve side effects from the notebook. Tables 5, 6, and 7 write `E1_estimator_reliability_results.csv`, `E2_noise_privacy_tradeoff_results.csv`, and `E3_high_alignment_metric_sweep_results.csv` in the current working directory. Table 8 writes a CSV and `.npz` histories under `E4_outputs_noLEAF/`. These are generated artifacts, not input files required to use the repository.

The Python conversion and source equality have been checked without running the training experiments. The author has confirmed that the original notebook was used directly for the paper experiments. This reorganization does not claim a new independent reproduction, exact historical Colab package versions, or numerical equivalence between the earlier edited table notebooks and the original cells.
