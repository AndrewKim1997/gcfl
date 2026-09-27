# GCFL

**Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It**  
Official research code · Transactions on Machine Learning Research (TMLR), 2026

[Paper (OpenReview)](https://openreview.net/forum?id=Ck3q5YdWIv) · [Camera-ready preprint](https://arxiv.org/abs/2509.02391) · [Quick start](#quick-start) · [Paper experiments](#paper-experiments) · [Reproducibility](docs/reproducibility.md) · [License](LICENSE)

GCFL studies federated learning as a strategic system: clients may game public metrics, free-ride, collude, or attack. The paper develops monitoring and audit-oriented indices, including manipulability, the price of gaming, and the price of cooperation. This repository provides the original experimental record, scripts extracted from it, and a small reusable interface for the stylized simulation.

## Method at a glance

The experiments compare a public metric with underlying welfare as client behavior and governance rules change. For aligned and gaming scenarios with positive aligned welfare, the price of gaming is `(W_aligned - W_gaming) / W_aligned`. Audits and penalties can change participation and the metric–welfare gap. See the [paper](https://openreview.net/forum?id=Ck3q5YdWIv) for the full framework and assumptions.

## Quick start

With Python 3.10 or newer, run from the repository root:

```bash
python -m pip install -e .
python examples/basic_usage.py
```

The [small example](examples/basic_usage.py) compares aligned and gaming scenarios with 20 clients and 30 rounds. It is illustrative and does not reproduce a paper table. [`src/gcfl/stylized.py`](src/gcfl/stylized.py) contains the simulation and PoG estimator copied from the original notebook, while [`src/gcfl/__init__.py`](src/gcfl/__init__.py) exposes the small public API.

## Paper experiments

Install the additional dependencies, then run each script from the repository root:

```bash
python -m pip install -e '.[experiments]'
python experiments/01_stylized_simulation.py      # Tables 1–3
python experiments/02_real_world_fl.py            # Table 4: Fashion-MNIST
python experiments/03_estimator_reliability.py    # Table 5: partial audits
python experiments/04_noise_and_auditability.py   # Table 6: noise and audits
python experiments/05_high_alignment_metrics.py  # Table 7: metric alignment
python experiments/06_modern_attack.py            # Table 8: FEMNIST
```

These scripts are exact text extracts of the paper's original experimental code cells. They retain the original progress messages and generated files. Fashion-MNIST and FEMNIST are downloaded on first use; federated runs may take substantial time. The [experiment-to-result and output notes](docs/reproducibility.md) give the mapping and file locations.

## Source and reproducibility

[`archive/GCFL_main.ipynb`](archive/GCFL_main.ipynb) is the unmodified notebook the author used for the paper experiments. [`tools/extract_experiments.py`](tools/extract_experiments.py) verifies its SHA-256 and regenerates the six standalone scripts and reusable stylized simulator from specified source cells. The Colab `!pip` setup cell is represented by the installation command above. To check that the committed code still matches the archived source:

```bash
python tools/extract_experiments.py --check
```

The previous, edited, table-formatted notebooks were removed from the current tree; they remain in the [pre-reorganization commit](https://github.com/AndrewKim1997/gcfl/tree/9ee77fefa1fcba734e37bcdb58641d30351336f2/notebooks). The [reproducibility notes](docs/reproducibility.md) state which checks were performed and what was not rerun.

## Citation

If you use this code, please cite the paper. [CITATION.cff](CITATION.cff) provides repository citation metadata.

```bibtex
@article{kim2026gaming,
  title={Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It},
  author={Dongseok Kim and Hyoungsun Choi and Mohamed Jismy Aashik Rasool and Gisung Oh},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=Ck3q5YdWIv}
}
```

## License

Released under the [MIT License](LICENSE). Downloaded datasets retain their providers' terms.
