# GCFL

**Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It**  
Official research code · Transactions on Machine Learning Research (TMLR), 2026

[Paper (OpenReview)](https://openreview.net/forum?id=Ck3q5YdWIv) · [Camera-ready preprint](https://arxiv.org/abs/2509.02391) · [Paper experiments](#paper-experiments) · [Reproducibility and provenance](docs/reproducibility.md) · [License](LICENSE)

GCFL studies federated learning as a strategic system in which clients can game public metrics, free-ride, collude, or attack. The paper introduces monitoring and audit-oriented indices, including manipulability, the price of gaming, and the price of cooperation.

## Quick start

With Python 3.10 or newer, run from the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/01_stylized_simulation.py
```

The stylized simulation is the simplest entry point. The other experiments involve federated training and can take substantially longer. The scripts are exact text extracts of the experimental code cells in the [original notebook](notebooks/original/GCFL_main.ipynb); they preserve its output and file-writing behavior, including progress messages where present.

## Paper experiments

Run a script from the repository root to execute its corresponding original experiment:

```bash
python experiments/01_stylized_simulation.py      # Tables 1–3
python experiments/02_real_world_fl.py            # Table 4: Fashion-MNIST
python experiments/03_estimator_reliability.py    # Table 5: partial audits
python experiments/04_noise_and_auditability.py   # Table 6: noise and audits
python experiments/05_high_alignment_metrics.py  # Table 7: metric alignment
python experiments/06_modern_attack.py            # Table 8: FEMNIST
```

The Fashion-MNIST experiments download the dataset on first use. The FEMNIST experiment uses Flower Datasets and Hugging Face datasets and needs network access. Some original cells write CSV files or histories to the working directory; see the [experiment notes](docs/reproducibility.md) before running them.

## Source and reproducibility

[`notebooks/original/GCFL_main.ipynb`](notebooks/original/GCFL_main.ipynb) is the author-provided notebook used for the paper experiments. It remains at its existing path. [`tools/extract_experiments.py`](tools/extract_experiments.py) verifies its SHA-256 and extracts six unmodified Python code cells into [`experiments/`](experiments/). The separate Colab `!pip install` cell is handled by the dependency installation command above. Check the committed scripts against their source cells with:

```bash
python tools/extract_experiments.py --check
```

The existing [`notebooks/`](notebooks/) files named by table are earlier, edited, table-formatted derivatives. They remain available for readers who prefer notebooks; the original notebook and its directly extracted scripts are the provenance reference. The [reproducibility notes](docs/reproducibility.md) describe the distinction and the scope of checks performed for this release.

## Citation

If you use the code, please cite the paper. [CITATION.cff](CITATION.cff) provides the repository citation metadata.

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
