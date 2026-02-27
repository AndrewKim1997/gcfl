<p align="center">
  <a href="https://arxiv.org/abs/2509.02391">
    <img alt="arXiv: 2509.02391"
         src="https://img.shields.io/badge/arXiv%3A%202509.02391-b31b1b?style=flat-square&logo=arXiv&logoColor=white&labelColor=b31b1b">
  </a>
  <a href="https://openreview.net/forum?id=Ck3q5YdWIv">
    <img alt="Journal: Transactions on Machine Learning Research"
         src="https://img.shields.io/badge/Transactions_on_Machine_Learning_Research-1e3787?style=flat-square&logo=tumblr&logoColor=white&labelColor=1e3787">
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
  <a href="#-quick-start">Quick start</a> •
  <a href="#-reproduce-paper-tables">Reproduce</a> •
  <a href="#-reproducibility-notes">Notes</a> •
  <a href="#-citation">Cite</a>
</p>

> Standalone notebooks that reproduce the paper tables for our “FL as a governed strategic system” view—separating welfare-improving behavior from metric gaming, and supporting monitoring via manipulability / price-of-gaming / price-of-cooperation indices and audit-oriented tooling.

- **Paper**: _Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It_
  - **TMLR**: https://openreview.net/forum?id=Ck3q5YdWIv
  - **arXiv**: https://arxiv.org/abs/2509.02391 
- **What this repo provides**: table-reproduction notebooks that print **only the corresponding table** to the console (no noisy intermediate logs).

---

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

## 🧪 Reproduce paper tables

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

## 🧷 Reproducibility notes

* Random seeds are fixed inside each notebook (see the `seed` field in the config).
* Notebooks are designed to avoid noisy intermediate logs and print table-ready summaries only.
* Large artifacts (raw datasets, checkpoints, long histories) are not committed to the repository.

---

## 📁 Repository structure (typical)

```
├── notebooks/
│   ├── Table_1_3(stylized_simulation).ipynb
│   ├── Table_4(real_world_FL).ipynb
│   ├── Table_5(estimator_reliability).ipynb
│   ├── Table_6(noise_and_auditability).ipynb
│   ├── Table_7(high_alignment_metrics).ipynb
│   └── Table_8(modern_attack).ipynb
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## 📚 Citation

If you use this code, please cite the paper and this repository.

* See `CITATION.cff` for citation metadata.

---

## 📝 License

This project is released under the terms of the license in `LICENSE`.
