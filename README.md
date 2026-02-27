# GCFL: Reproducibility Code

This repository provides the reproducibility package for the paper:

**Gaming and Cooperation in Federated Learning: What Can Happen and How to Monitor It**

It contains standalone notebooks that reproduce the paper tables. Each script prints **only the corresponding table** to the console.

---

## Contents

### Table reproduction notebooks

| Paper artifact | Script |
|---|---|
| Tables 1–3 (stylized simulation) | `notebooks/Table_1_3(stylized_simulation).ipynb` |
| Table 4 (Fashion-MNIST FL: head metric vs tail welfare) | `notebooks/Table_4(real_world_FL).ipynb` |
| Table 5 (E1: estimator reliability under audits) | `notebooks/Table_5(estimator_reliability).ipynb` |
| Table 6 (E2: noise/privacy trade-off) | `notebooks/Table_6(noise_and_auditability).ipynb` |
| Table 7 (E3: high-alignment metric sweep) | `notebooks/Table_7(high_alignment_metrics).ipynb` |
| Table 8 (E4: FEMNIST modern attack–defense replication) | `notebooks/Table_8(modern_attack).ipynb` |

---

## Setup

### Requirements
- Python 3.10+ recommended
- `pip` environment

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## Reproducing the tables

Run the script(s) for the table(s) you need:

```bash
python notebooks/Table_1_3(stylized_simulation).ipynb
python notebooks/Table_4(real_world_FL).ipynb
python notebooks/Table_5(estimator_reliability).ipynb
python notebooks/Table_6(noise_and_auditability).ipynb
python notebooks/Table_7(high_alignment_metrics).ipynb
python notebooks/Table_8(modern_attack).ipynb
```

Notes:

* **Tables 1–7** use Fashion-MNIST or stylized simulations and should run after installing `requirements.txt`.
* **Table 8** additionally downloads FEMNIST via `flwr-datasets`/`datasets` at runtime.

---

## Reproducibility notes

* Random seeds are fixed inside each script (see the `seed` field in the config).
* The notebooks are designed to avoid noisy intermediate logs and print table-ready summaries only.
* Large artifacts (raw datasets, checkpoints, long histories) are not committed to the repository.

---

## Repository structure (typical)

```
.
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

## Citation

If you use this code, please cite the paper and this repository.

* See `CITATION.cff` for citation metadata.
