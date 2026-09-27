import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms


# ============================================================
# E1: Estimator Reliability (Ground-truth vs Audit-based estimator)
# Standalone code (not meant to be merged into your old script)
# ============================================================

# ----------------------------
# 0) Config / Seed
# ----------------------------
@dataclass
class Config:
    # Keep small for Colab
    n_clients: int = 12
    n_rounds: int = 25
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9

    # Non-IID partition
    dirichlet_alpha: float = 0.5

    # Head/Tail split
    head_classes: Tuple[int, ...] = (0, 1, 2, 3, 4)
    tail_classes: Tuple[int, ...] = (5, 6, 7, 8, 9)

    # Public leak
    leak_fraction: float = 1.0  # use all head-only public val for max gaming effect

    # Strategy mix for profiles
    gaming_fracs: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)     # for GT / estimator comparison
    benign_frac: float = 0.10                               # small benign cooperation fraction
    update_scale_factor: float = 1.75                       # for update-scaling gaming

    # Audit budgets to test estimator reliability
    audit_budgets: Tuple[float, ...] = (0.10, 0.25, 0.50)
    audit_trials: int = 5  # resample audits to estimate estimator noise

    # Evaluation
    tail_k: int = 7          # tail-mean over last K rounds
    pog_threshold: float = 0.05  # risk threshold for FP/FN

    seed: int = 42
    root: str = "./data"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# ----------------------------
# 1) Dataset preparation
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = datasets.FashionMNIST(root=CFG.root, train=True, download=True, transform=transform)

# We'll use:
# - 50k for client-local data (partitioned)
# - 10k as public validation candidates (from the same training set)
train_local_size = 50_000
idx_all = np.arange(len(full_train))
idx_local = idx_all[:train_local_size]
idx_public_candidates = idx_all[train_local_size:]

train_local_set = Subset(full_train, idx_local)

targets_all = np.array(full_train.targets)

# Public validation set: head-only from last 10k
head_mask = np.isin(targets_all, list(CFG.head_classes))
public_head_indices = [i for i in idx_public_candidates if head_mask[i]]
public_head_indices = np.array(public_head_indices)

rng = np.random.default_rng(CFG.seed)
rng.shuffle(public_head_indices)

leak_size = int(CFG.leak_fraction * len(public_head_indices))
leak_indices = public_head_indices[:leak_size]

public_val_set = Subset(full_train, public_head_indices)
public_val_loader = DataLoader(public_val_set, batch_size=CFG.batch_size, shuffle=False)

# Leak subset that gaming clients can append
leak_subset = Subset(full_train, leak_indices)


# ----------------------------
# 2) Utilities: partition + filtering
# ----------------------------
def make_dirichlet_partitions(labels: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator):
    """Returns list of index arrays per client (indices w.r.t. train_local_set)."""
    n_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    class_indices = []
    for k in range(n_classes):
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        class_indices.append(idx_k)

    for k in range(n_classes):
        idx_k = class_indices[k]
        n_k = len(idx_k)
        if n_k == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * n_k).astype(int)
        shards = np.split(idx_k, splits[:-1])
        for cid, shard in enumerate(shards):
            client_indices[cid].extend(shard.tolist())

    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


def filter_subset_by_labels(base_subset: Subset, allowed_labels: Tuple[int, ...]) -> Subset:
    """base_subset is a Subset of train_local_set; keep only items with labels in allowed_labels."""
    allowed = set(allowed_labels)
    kept = []
    # base_subset.indices are indices inside train_local_set (0..train_local_size-1)
    for idx_in_local in base_subset.indices:
        global_idx = train_local_set.indices[idx_in_local]  # index into full_train
        y = int(full_train.targets[global_idx])
        if y in allowed:
            kept.append(idx_in_local)
    return Subset(train_local_set, kept)


def split_subset_train_eval(base_subset: Subset, eval_ratio: float = 0.2) -> Tuple[Subset, Subset]:
    """Split a client's local subset into train/eval (by index)."""
    idxs = list(base_subset.indices)
    rng_local = np.random.default_rng(CFG.seed + 123)
    rng_local.shuffle(idxs)
    n_eval = int(round(eval_ratio * len(idxs)))
    eval_idxs = idxs[:n_eval]
    train_idxs = idxs[n_eval:]
    return Subset(train_local_set, train_idxs), Subset(train_local_set, eval_idxs)


def tail_only_subset(base_subset: Subset) -> Subset:
    return filter_subset_by_labels(base_subset, CFG.tail_classes)


def oversample_tail(train_subset: Subset, factor: int = 2) -> ConcatDataset:
    """Benign cooperation: upweight tail samples in local training."""
    tail_sub = tail_only_subset(train_subset)
    # If tail is empty, just return original
    if len(tail_sub) == 0:
        return ConcatDataset([train_subset])
    reps = [tail_sub for _ in range(factor)]
    return ConcatDataset([train_subset] + reps)


# Build base client datasets (Dirichlet)
train_targets_local = np.array(full_train.targets)[idx_local]
client_indices = make_dirichlet_partitions(
    labels=train_targets_local,
    n_clients=CFG.n_clients,
    alpha=CFG.dirichlet_alpha,
    rng=np.random.default_rng(CFG.seed)
)

client_base_subsets = [Subset(train_local_set, idxs) for idxs in client_indices]

# For auditing/welfare evaluation, each client has a held-out local eval set; welfare = tail-accuracy on that eval set.
client_train_subsets: List[Subset] = []
client_eval_subsets: List[Subset] = []
client_tail_eval_subsets: List[Subset] = []

for cid in range(CFG.n_clients):
    tr, ev = split_subset_train_eval(client_base_subsets[cid], eval_ratio=0.2)
    client_train_subsets.append(tr)
    client_eval_subsets.append(ev)
    client_tail_eval_subsets.append(tail_only_subset(ev))


# ----------------------------
# 3) Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model() -> nn.Module:
    return SimpleCNN().to(CFG.device)


# ----------------------------
# 4) Train/Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return float(correct / total) if total > 0 else 0.0


def local_train(model: nn.Module, dataset, epochs: int) -> nn.Module:
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    opt = optim.SGD(model.parameters(), lr=CFG.lr, momentum=CFG.momentum)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(CFG.device), y.to(CFG.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return model


def state_dict_add_scaled_delta(global_sd: Dict[str, torch.Tensor],
                                local_sd: Dict[str, torch.Tensor],
                                scale: float) -> Dict[str, torch.Tensor]:
    """Return transmitted state = global + scale*(local - global)."""
    out = {}
    for k in global_sd.keys():
        out[k] = global_sd[k] + scale * (local_sd[k] - global_sd[k])
    return out


def fedavg_from_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average a list of state_dicts (FedAvg)."""
    avg = {}
    for k in states[0].keys():
        stacked = torch.stack([sd[k] for sd in states], dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


# ----------------------------
# 5) Strategy definitions
# ----------------------------
STR_HONEST = "HONEST"
STR_BENIGN = "BENIGN_TAIL_UPWEIGHT"
STR_GAME_HEAD = "GAMING_HEAD_ONLY_LEAK"
STR_GAME_SCALE = "GAMING_UPDATE_SCALING"


def assign_strategies(n_clients: int,
                      gaming_frac: float,
                      benign_frac: float,
                      rng: np.random.Generator) -> Dict[int, str]:
    """
    Assign each client a strategy from {HONEST, BENIGN, GAMING_HEAD, GAMING_SCALE}.
    To keep it simple:
      - benign_frac portion get BENIGN
      - gaming_frac portion get a GAMING strategy (half head-only, half update-scaling)
      - rest are HONEST
    """
    all_ids = np.arange(n_clients)
    rng.shuffle(all_ids)

    n_benign = int(round(benign_frac * n_clients))
    n_gaming = int(round(gaming_frac * n_clients))

    benign_ids = set(all_ids[:n_benign])
    gaming_ids = list(all_ids[n_benign:n_benign + n_gaming])
    honest_ids = set(all_ids[n_benign + n_gaming:])

    strat = {}
    for cid in benign_ids:
        strat[cid] = STR_BENIGN
    for cid in honest_ids:
        strat[cid] = STR_HONEST

    # split gaming IDs into two types
    half = len(gaming_ids) // 2
    for cid in gaming_ids[:half]:
        strat[cid] = STR_GAME_HEAD
    for cid in gaming_ids[half:]:
        strat[cid] = STR_GAME_SCALE

    return strat


def build_client_train_dataset(cid: int, strategy: str):
    base_train = client_train_subsets[cid]

    if strategy == STR_HONEST:
        return base_train

    if strategy == STR_BENIGN:
        # upweight tail a bit (benign cooperation)
        return oversample_tail(base_train, factor=2)

    if strategy == STR_GAME_HEAD:
        # Remove tail samples from local training; append public head leak
        head_only = filter_subset_by_labels(base_train, CFG.head_classes)
        return ConcatDataset([head_only, leak_subset])

    if strategy == STR_GAME_SCALE:
        # Training data itself is honest; manipulation happens in transmitted update
        return base_train

    raise ValueError(f"Unknown strategy: {strategy}")


# ----------------------------
# 6) Run FL once per profile, store per-round:
#    - public metric M_t (head-only public val accuracy)
#    - per-client tail accuracies on client-heldout tail eval (for GT + audit sampling)
# ----------------------------
def run_profile_once(gaming_frac: float, benign_frac: float, seed_offset: int = 0):
    rng = np.random.default_rng(CFG.seed + seed_offset)

    strat_map = assign_strategies(CFG.n_clients, gaming_frac, benign_frac, rng)

    # Pre-build each client's training dataset according to strategy
    client_train_ds = [build_client_train_dataset(cid, strat_map[cid]) for cid in range(CFG.n_clients)]

    global_model = get_model()
    global_sd = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    M_hist: List[float] = []
    tail_acc_matrix = np.zeros((CFG.n_rounds, CFG.n_clients), dtype=np.float32)

    for rnd in range(CFG.n_rounds):
        transmitted_states: List[Dict[str, torch.Tensor]] = []

        for cid in range(CFG.n_clients):
            local_model = get_model()
            local_model.load_state_dict(global_sd)

            local_model = local_train(local_model, client_train_ds[cid], epochs=CFG.local_epochs)
            local_sd = local_model.state_dict()

            # Manipulation at transmission time for UPDATE_SCALING gaming
            if strat_map[cid] == STR_GAME_SCALE and gaming_frac > 0:
                tx_sd = state_dict_add_scaled_delta(global_sd, local_sd, scale=CFG.update_scale_factor)
            else:
                tx_sd = {k: v.detach().clone() for k, v in local_sd.items()}

            transmitted_states.append(tx_sd)

        # FedAvg update
        global_sd = fedavg_from_states(transmitted_states)
        global_model.load_state_dict(global_sd)

        # Metric: server-observable public head validation accuracy
        M_t = evaluate_acc(global_model, public_val_loader)
        M_hist.append(M_t)

        # Welfare ingredient: per-client tail eval accuracy (experimenter can compute for GT;
        # audit will subsample from this vector)
        for cid in range(CFG.n_clients):
            tail_ev = client_tail_eval_subsets[cid]
            if len(tail_ev) == 0:
                tail_acc_matrix[rnd, cid] = np.nan  # if a client has no tail samples
                continue
            loader = DataLoader(tail_ev, batch_size=CFG.batch_size, shuffle=False)
            tail_acc_matrix[rnd, cid] = evaluate_acc(global_model, loader)

        if (rnd + 1) % 10 == 0 or rnd == 0:
            w_full = np.nanmean(tail_acc_matrix[rnd])
            print(f"[Profile gf={gaming_frac:.2f}] Round {rnd+1:>2}/{CFG.n_rounds} | "
                  f"M(head public)={M_t:.4f} | W_full_tail(mean clients)={w_full:.4f}")

    return {
        "gaming_frac": gaming_frac,
        "benign_frac": benign_frac,
        "strategy_map": strat_map,
        "M_hist": np.array(M_hist, dtype=np.float32),
        "tail_acc_matrix": tail_acc_matrix,  # shape [T, N]
    }


def tail_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return float("nan")
    if len(x) <= k:
        return float(np.nanmean(x))
    return float(np.nanmean(x[-k:]))


# ----------------------------
# 7) Estimator: audit-based welfare estimate from tail acc matrix
#    - GT welfare per round: mean over all clients (nanmean)
#    - Estimated welfare per round: mean over audited clients (nanmean over sampled subset)
#    - PoG uses baseline welfare from aligned profile
# ----------------------------
def compute_pog_series(w_ref: float, w_series: np.ndarray) -> np.ndarray:
    # PoG_t = (W_ref - W_t) / W_ref
    if not np.isfinite(w_ref) or w_ref <= 1e-8:
        return np.full_like(w_series, np.nan, dtype=np.float32)
    out = (w_ref - w_series) / w_ref
    return out.astype(np.float32)


def audit_estimate_welfare_series(tail_acc_matrix: np.ndarray,
                                 audit_budget: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    tail_acc_matrix: [T, N] per-client tail acc (nan if no tail data)
    returns: W_hat_series [T]
    """
    T, N = tail_acc_matrix.shape
    m = max(1, int(round(audit_budget * N)))
    W_hat = np.zeros(T, dtype=np.float32)
    for t in range(T):
        audited = rng.choice(np.arange(N), size=m, replace=False)
        W_hat[t] = float(np.nanmean(tail_acc_matrix[t, audited]))
    return W_hat


def detection_delay(pog_hat: np.ndarray, tau: float) -> int:
    """First round index (1-based) where PoG_hat >= tau; return 0 if never detected."""
    for i, v in enumerate(pog_hat):
        if np.isfinite(v) and v >= tau:
            return i + 1
    return 0


# ----------------------------
# 8) Main: run baseline + profiles, then evaluate estimator reliability
# ----------------------------
def main():
    print("\n=== E1: Estimator Reliability (GT vs Audit-based Estimator) ===")
    print(f"Device: {CFG.device}")
    print(f"Clients={CFG.n_clients}, Rounds={CFG.n_rounds}, LocalEpochs={CFG.local_epochs}")
    print(f"Audit budgets={CFG.audit_budgets}, trials={CFG.audit_trials}\n")

    # 8.1 Baseline aligned run (gaming_frac=0.0), used as welfare reference
    baseline = run_profile_once(gaming_frac=0.0, benign_frac=CFG.benign_frac, seed_offset=0)
    M_ref_series = baseline["M_hist"]
    W_ref_series_full = np.nanmean(baseline["tail_acc_matrix"], axis=1)  # GT welfare series
    W_ref = tail_mean(W_ref_series_full, CFG.tail_k)
    M_ref = tail_mean(M_ref_series, CFG.tail_k)

    print("\n[Baseline] Reference values (tail-mean over last K rounds)")
    print(f"  W_ref (tail welfare) = {W_ref:.4f}")
    print(f"  M_ref (head metric)  = {M_ref:.4f}\n")

    # 8.2 Run profiles for each gaming_frac (one run each), store GT
    profiles = []
    seed_offset = 10
    for gf in CFG.gaming_fracs:
        prof = run_profile_once(gaming_frac=gf, benign_frac=CFG.benign_frac, seed_offset=seed_offset)
        seed_offset += 1
        profiles.append(prof)

    # 8.3 For each profile, compute GT PoG and estimator performance for each audit budget
    rows = []
    for prof in profiles:
        gf = prof["gaming_frac"]
        M_series = prof["M_hist"]
        tail_mat = prof["tail_acc_matrix"]

        # Ground-truth welfare series (full visibility, experimenter only)
        W_full_series = np.nanmean(tail_mat, axis=1)
        W_full = tail_mean(W_full_series, CFG.tail_k)

        # Ground-truth PoG (final)
        pog_gt_series = compute_pog_series(W_ref, W_full_series)
        pog_gt = tail_mean(pog_gt_series, CFG.tail_k)

        # Simple manipulation index proxy: metric inflation vs baseline (final)
        M_final = tail_mean(M_series, CFG.tail_k)
        manip_delta = M_final - M_ref

        # Risk label for FP/FN (based on GT)
        is_risky_gt = (np.isfinite(pog_gt) and pog_gt >= CFG.pog_threshold)

        print(f"\n[Profile gf={gf:.2f}] GT summary:")
        print(f"  M_final(head metric)  = {M_final:.4f} (delta vs ref {manip_delta:+.4f})")
        print(f"  W_full(tail welfare)  = {W_full:.4f}")
        print(f"  PoG_GT                = {pog_gt:.4f} (risk>=tau? {is_risky_gt})")

        # Estimator: for each budget, resample audits multiple times
        for b in CFG.audit_budgets:
            pog_hats = []
            delays = []
            risky_preds = []

            for tr in range(CFG.audit_trials):
                rng_a = np.random.default_rng(CFG.seed + 1000 + tr + int(100 * b) + int(1000 * gf))
                W_hat_series = audit_estimate_welfare_series(tail_mat, audit_budget=b, rng=rng_a)
                pog_hat_series = compute_pog_series(W_ref, W_hat_series)
                pog_hat = tail_mean(pog_hat_series, CFG.tail_k)

                pog_hats.append(pog_hat)
                delays.append(detection_delay(pog_hat_series, CFG.pog_threshold))
                risky_preds.append(np.isfinite(pog_hat) and pog_hat >= CFG.pog_threshold)

            pog_hat_mean = float(np.nanmean(pog_hats))
            pog_hat_std = float(np.nanstd(pog_hats))
            delay_mean = float(np.mean(delays))

            # Classification outcomes
            pred_risky = (np.mean(risky_preds) >= 0.5)  # majority vote across trials
            fp = int((not is_risky_gt) and pred_risky)
            fn = int(is_risky_gt and (not pred_risky))

            rows.append({
                "gaming_frac": gf,
                "audit_budget": b,
                "M_final": M_final,
                "M_delta_vs_ref": manip_delta,
                "W_full": W_full,
                "PoG_GT": pog_gt,
                "PoG_hat_mean": pog_hat_mean,
                "PoG_hat_std": pog_hat_std,
                "delay_mean_rounds": delay_mean,
                "risk_GT": int(is_risky_gt),
                "risk_pred_majority": int(pred_risky),
                "FP": fp,
                "FN": fn,
            })

            print(f"  [Audit b={b:.2f}] PoG_hat={pog_hat_mean:.4f}±{pog_hat_std:.4f} | "
                  f"delay~{delay_mean:.1f} rounds | pred_risky={pred_risky} | FP={fp} FN={fn}")

    # 8.4 Aggregate reliability metrics across profiles for each budget
    print("\n=== Aggregated Estimator Reliability (across profiles) ===")
    rows_by_b = {}
    for r in rows:
        rows_by_b.setdefault(r["audit_budget"], []).append(r)

    for b, rs in rows_by_b.items():
        gt = np.array([x["PoG_GT"] for x in rs], dtype=np.float32)
        hat = np.array([x["PoG_hat_mean"] for x in rs], dtype=np.float32)

        # Spearman rank correlation (manual, simple)
        def rankdata(a):
            temp = a.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(a))
            return ranks.astype(np.float32)

        mask = np.isfinite(gt) & np.isfinite(hat)
        if mask.sum() >= 2:
            rg = rankdata(gt[mask])
            rh = rankdata(hat[mask])
            spearman = float(np.corrcoef(rg, rh)[0, 1])
        else:
            spearman = float("nan")

        FP = sum(x["FP"] for x in rs)
        FN = sum(x["FN"] for x in rs)
        n = len(rs)

        print(f"Audit b={b:.2f} | Spearman(PoG_GT, PoG_hat)={spearman:.3f} | "
              f"FP={FP}/{n} FN={FN}/{n}")

    # 8.5 Save CSV
    out_path = "E1_estimator_reliability_results.csv"
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()