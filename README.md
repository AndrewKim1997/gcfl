<p align="center">
  <a href="https://openreview.net/forum?id=Ck3q5YdWIv">
    <img alt="Journal: Transactions on Machine Learning Research"
         src="https://img.shields.io/badge/Transactions_on_Machine_Learning_Research-1e3787?style=flat-square&logo=tumblr&logoColor=white&labelColor=1e3787">
  </a>
  <a href="https://arxiv.org/abs/2509.02391">
    <img alt="arXiv: 2509.02391"
         src="https://img.shields.io/badge/arXiv%3A%202509.02391-b31b1b?style=flat-square&logo=arXiv&logoColor=white&labelColor=b31b1b">
  </a>
  <a href="https://colab.research.google.com">
    <img alt="Google Colab"
       src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white&labelColor=F9AB00">
  </a>
  <img alt="Python 3.10+"
       src="https://img.shields.io/badge/Python%203.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=3776AB">
  <a href="LICENSE">
    <img alt="License"
         src="https://img.shields.io/badge/License-See%20LICENSE-16a34a?style=flat-square&logo=opensourceinitiative&logoColor=white&labelColor=16a34a">
  </a>
</p>

<h1 align="center">GCFL: Reproducibility Code</h1>

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#reproduce-paper-tables">Reproduce</a> •
  <a href="#running-on-google-colab-optional">Colab</a> •
  <a href="#reproducibility-notes">Notes</a> •
  <a href="#citation">Cite</a>
</p>

> This repository reproduces the paper’s core tables for **Gaming and Cooperation in Federated Learning**, which studies federated learning as a **strategic, governed system**: clients may **game metrics**, **free-ride**, **collude**, or **attack**, and these behaviors can distort reported progress and reduce welfare.  

> The paper proposes **monitoring- and audit-oriented diagnostics** (e.g., manipulability / price-of-gaming / price-of-cooperation style indices) to separate **welfare-improving** updates from **strategic** ones, and evaluates them in stylized simulations and modern FL benchmarks.  

> Each notebook in this repo is **standalone** and prints **only the corresponding paper table** (minimal logs) for reproducibility.

- **Paper**: _Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It_
  - **TMLR**: https://openreview.net/forum?id=Ck3q5YdWIv
  - **arXiv**: https://arxiv.org/abs/2509.02391
- **What this repo provides**: table-reproduction notebooks that print **only the corresponding table** to the console (no noisy intermediate logs).
- **Supplementary notebook (original)**: `notebooks/original/GCFL_main.ipynb` is the original supplementary material notebook uploaded to OpenReview (kept here for reference).

---

<a id="quick-start"></a>
## ⚡ Quick start

### Setup

```bash
pip install -r requirements.txt
````

### Run a notebook from the terminal (recommended)

Notebooks are executed by converting to a temporary Python script and piping to `python`, so printed tables appear in your terminal:

```bash
jupyter nbconvert --to python --stdout "notebooks/Table_4(real_world_FL).ipynb" | python -
```

> Note: filenames include parentheses, so **quotes are required** in most shells.

---

<a id="reproduce-paper-tables"></a>

## ⟲ Reproduce paper tables

Each notebook is standalone and prints **only** its table.

```bash
# Tables 1–3 (stylized simulation)
jupyter nbconvert --to python --stdout "notebooks/Table_1_3(stylized_simulation).ipynb" | python -

# Table 4 (Fashion-MNIST FL: head metric vs tail welfare)
jupyter nbconvert --to python --stdout "notebooks/Table_4(real_world_FL).ipynb" | python -

# Table 5 (E1: estimator reliability under audits)
jupyter nbconvert --to python --stdout "notebooks/Table_5(estimator_reliability).ipynb" | python -

# Table 6 (E2: noise/privacy trade-off)
jupyter nbconvert --to python --stdout "notebooks/Table_6(noise_and_auditability).ipynb" | python -

# Table 7 (E3: high-alignment metric sweep)
jupyter nbconvert --to python --stdout "notebooks/Table_7(high_alignment_metrics).ipynb" | python -

# Table 8 (E4: FEMNIST modern attack–defense replication)
jupyter nbconvert --to python --stdout "notebooks/Table_8(modern_attack).ipynb" | python -
```

**Notes**

* **Tables 1–7** use Fashion-MNIST or stylized simulations and should run after installing `requirements.txt`.
* **Table 8** additionally downloads FEMNIST via `flwr-datasets` / `datasets` at runtime.

---

<a id="running-on-google-colab-optional"></a>

## ☁ Running on Google Colab (optional)

You can also run the notebooks on **Google Colab** after downloading/cloning this repository.

1. Clone (or upload) this repository into your Colab environment.
2. Install dependencies.
3. Open any notebook under `notebooks/` and run all cells.

### Minimal Colab setup cell

```python
# In a Colab notebook cell
!git clone https://github.com/AndrewKim1997/gcfl.git
%cd gcfl
!pip install -q -r requirements.txt
```

> We do not provide a hosted Colab runtime; the notebooks run in your own Colab environment after cloning/downloading this repo.

---

<a id="reproducibility-notes"></a>

## ✎ Reproducibility notes

* Random seeds are fixed inside each notebook (see the `seed` field in the config).
* Notebooks are designed to avoid noisy intermediate logs and print table-ready summaries only.
* Large artifacts (raw datasets, checkpoints, long histories) are not committed to the repository.

---

## ▦ Repository structure (typical)

```
├── notebooks/
│   ├── Table_1_3(stylized_simulation).ipynb
│   ├── Table_4(real_world_FL).ipynb
│   ├── Table_5(estimator_reliability).ipynb
│   ├── Table_6(noise_and_auditability).ipynb
│   ├── Table_7(high_alignment_metrics).ipynb
│   ├── Table_8(modern_attack).ipynb
│   └── original/
│       └── GCFL_main.ipynb
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

<a id="citation"></a>

## ❖ Citation

If you use this code, please cite the paper and this repository.

* See `CITATION.cff` for citation metadata.

---

## ⚖ License

This project is released under the terms of the license in `LICENSE`.

```
```
