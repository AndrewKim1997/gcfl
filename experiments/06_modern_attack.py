# ============================================================
# E4 FULL (No LEAF): FEMNIST via HuggingFace/Flower Datasets
#   - Attacks: PoisonedFL (multi-round consistency + proximal + dyn magnitude),
#              Backdoor/Model Replacement
#   - Defenses: FedCC (linear CKA filtering), Attack-Adaptive Aggregation
#   - FL: FedAvg skeleton (with defense hooks)
#   - Output: CSV + NPZ histories
# ============================================================

# If running on Colab, install once:
# !pip -q install flwr-datasets[vision] datasets

import os
import csv
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


# ----------------------------
# 0) Config / Seed
# ----------------------------
from dataclasses import dataclass
from typing import Tuple
import torch
import os

@dataclass
class Config:
    # compute budget (Colab-friendly)
    n_clients: int = 32            # CHANGED (24 -> 32)
    n_rounds: int = 100            # CHANGED (60 -> 100)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 0.0

    # participation
    clients_per_round: int = 12    # CHANGED (8 -> 12)

    # FEMNIST classes: typically 62 classes (digits + letters)
    # We'll define "head" as digits, "tail" as letters for head/tail gap.
    head_classes: Tuple[int, ...] = tuple(range(0, 10))       # digits
    tail_classes: Tuple[int, ...] = tuple(range(10, 62))      # letters

    # public sets (sampled from client train data)
    public_support_size: int = 128       # (CKA용) 유지
    public_head_eval_size: int = 4096    # CHANGED (2048 -> 4096)
    public_tail_eval_size: int = 4096    # CHANGED (2048 -> 4096)

    # malicious population + attack schedule
    malicious_frac: float = 0.30         # CHANGED (0.20 -> 0.30)
    attack_start_round: int = 5          # CHANGED (8 -> 5)

    # PoisonedFL-ish knobs (more faithful but still light)
    poisonedfl_scale: float = 2.2
    poisonedfl_scale_min: float = 1.0
    poisonedfl_scale_max: float = 4.0
    poisonedfl_dyn_eta: float = 0.8
    poisonedfl_beta_consistency: float = 1e-3
    poisonedfl_mu_global: float = 5e-4
    poisonedfl_tail_label_flip: bool = True

    # Backdoor / Model Replacement knobs
    backdoor_target_label: int = 0
    backdoor_poison_frac: float = 0.30
    model_replacement_gamma: float = 6.0
    trigger_size: int = 3
    trigger_value: float = 1.0   # in normalized space, clamped to [-1, 1]

    # FedCC (CKA) filtering knobs
    fedcc_reject_frac: float = 0.25
    fedcc_min_keep: int = 3

    # Attack-adaptive aggregation knobs
    adaagg_alpha: float = 3.0

    # summary / alarm
    tail_k: int = 10
    warmup_ignore: int = 5
    alarm_threshold_std: float = 2.0     # CHANGED (3.0 -> 2.0)

    # FEMNIST client selection safeguards
    min_train_samples_per_client: int = 40
    min_test_samples_per_client: int = 10
    max_partition_id_try: int = 5000

    # reproducibility
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # outputs
    out_dir: str = "./E4_outputs_noLEAF"
    csv_name: str = "E4_FEMNIST_2x2_results.csv"


CFG = Config()
os.makedirs(CFG.out_dir, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


# ----------------------------
# 1) FEMNIST via HF/Flower Datasets
# ----------------------------
def build_femnist_clients_from_hf(
    n_clients: int,
    train_ratio: float,
    seed: int,
    min_train: int,
    min_test: int,
    max_try: int,
) -> Tuple[List[TensorDataset], List[TensorDataset], int]:
    """
    Uses flwr_datasets FederatedDataset with NaturalIdPartitioner(partition_by="writer_id").
    Loads partitions sequentially (partition_id=0,1,2,...) until collecting enough clients.

    Returns:
      client_train_sets, client_test_sets, num_classes (assumed 62 if present; otherwise inferred)
    """
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner

    rng = np.random.default_rng(seed)

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
    )

    client_train_sets: List[TensorDataset] = []
    client_test_sets: List[TensorDataset] = []
    max_label = -1

    # Helper: PIL->torch normalized [-1,1]
    def to_tensors(part) -> Tuple[torch.Tensor, torch.Tensor]:
        images = part["image"]  # PIL images
        labels = np.array(part["character"], dtype=np.int64)
        # PIL -> np
        X = np.stack([np.array(img, dtype=np.float32) for img in images], axis=0)  # [N,28,28]
        X = X / 255.0
        X = (X - 0.5) / 0.5
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N,1,28,28]
        Y = torch.tensor(labels, dtype=torch.long)
        return X, Y

    for pid in range(max_try):
        if len(client_train_sets) >= n_clients:
            break
        try:
            part = fds.load_partition(partition_id=pid, split="train")
        except Exception:
            # Ran out of partitions or unsupported id
            break

        # Convert
        X, Y = to_tensors(part)
        n = len(Y)
        if n < (min_train + min_test):
            continue  # too small client

        # split per-client
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(round(train_ratio * n))
        tr_idx = idx[:cut]
        te_idx = idx[cut:]

        if len(tr_idx) < min_train or len(te_idx) < min_test:
            continue

        ds_tr = TensorDataset(X[tr_idx], Y[tr_idx])
        ds_te = TensorDataset(X[te_idx], Y[te_idx])

        client_train_sets.append(ds_tr)
        client_test_sets.append(ds_te)

        if n > 0:
            max_label = max(max_label, int(Y.max().item()))

    if len(client_train_sets) < n_clients:
        raise RuntimeError(
            f"Not enough FEMNIST clients collected. Got {len(client_train_sets)} / {n_clients}. "
            f"Increase max_partition_id_try (currently {max_try}) or relax min_*_samples."
        )

    num_classes = max_label + 1
    # FEMNIST is usually 62 classes; we keep inferred value to be safe.
    return client_train_sets, client_test_sets, num_classes


def filter_by_labels(ds: TensorDataset, allowed: Tuple[int, ...]) -> TensorDataset:
    allowed_set = set(int(x) for x in allowed)
    X, Y = ds.tensors
    if len(Y) == 0:
        return TensorDataset(X, Y)
    mask = torch.zeros_like(Y, dtype=torch.bool)
    for c in allowed_set:
        mask |= (Y == c)
    return TensorDataset(X[mask], Y[mask])


def sample_public_from_clients(
    client_train_sets: List[TensorDataset],
    allowed_labels: Tuple[int, ...],
    total_size: int,
    rng: np.random.Generator,
) -> TensorDataset:
    xs, ys = [], []
    for ds in client_train_sets:
        ds_f = filter_by_labels(ds, allowed_labels)
        if len(ds_f) == 0:
            continue
        X, Y = ds_f.tensors
        xs.append(X)
        ys.append(Y)
    if len(xs) == 0:
        return TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    n = len(Y)
    if n == 0:
        return TensorDataset(torch.empty(0), torch.empty(0, dtype=torch.long))
    idx = np.arange(n)
    rng.shuffle(idx)
    idx = idx[: min(total_size, n)]
    return TensorDataset(X[idx], Y[idx])


# ----------------------------
# 2) Model with penultimate features
# ----------------------------
class SimpleCNNFeat(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_feat: bool = False):
        z = self.features(x).flatten(1)
        feat = self.relu(self.fc1(z))
        logits = self.fc2(feat)
        if return_feat:
            return logits, feat
        return logits


def get_model(num_classes: int) -> nn.Module:
    return SimpleCNNFeat(num_classes=num_classes).to(CFG.device)


# ----------------------------
# 3) Eval helpers
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


@torch.no_grad()
def collect_features(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    model.eval()
    feats = []
    for x, _ in loader:
        x = x.to(CFG.device)
        _, f = model(x, return_feat=True)
        feats.append(f.detach())
    if len(feats) == 0:
        return torch.empty(0, 128, device=CFG.device)
    return torch.cat(feats, dim=0)


# ----------------------------
# 4) State helpers
# ----------------------------
def fedavg_mean(states: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = [1.0] * len(states)
    wsum = float(sum(weights))
    avg = {}
    for k in states[0].keys():
        stacked = torch.stack([sd[k] * float(w) for sd, w in zip(states, weights)], dim=0)
        avg[k] = stacked.sum(dim=0) / wsum
    return avg


def make_transmitted_state(global_sd: Dict[str, torch.Tensor], local_sd: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    tx = {}
    for k in global_sd.keys():
        tx[k] = global_sd[k] + scale * (local_sd[k] - global_sd[k])
    return tx


def flatten_update(global_sd: Dict[str, torch.Tensor], tx_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    vecs = []
    for k in global_sd.keys():
        vecs.append((tx_sd[k] - global_sd[k]).detach().flatten())
    return torch.cat(vecs, dim=0)


# ----------------------------
# 5) Backdoor wrapper (trigger + target relabel)
# ----------------------------
class BackdoorWrapper(Dataset):
    def __init__(self, base: TensorDataset, poison_frac: float, target_label: int,
                 trigger_size: int, trigger_value: float, seed: int):
        self.base = base
        self.poison_frac = float(poison_frac)
        self.target_label = int(target_label)
        self.trigger_size = int(trigger_size)
        self.trigger_value = float(trigger_value)
        rng = np.random.default_rng(seed)

        n = len(base)
        idx = np.arange(n)
        rng.shuffle(idx)
        self.poison_idx = set(idx[: int(round(self.poison_frac * n))].tolist())

    def __len__(self):
        return len(self.base)

    def _apply_trigger(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.clone()
        s = self.trigger_size
        x2[:, -s:, -s:] = torch.clamp(torch.tensor(self.trigger_value, dtype=x2.dtype), -1.0, 1.0)
        return x2

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if idx in self.poison_idx:
            x = self._apply_trigger(x)
            y = torch.tensor(self.target_label, dtype=torch.long)
        return x, y


# ----------------------------
# 6) PoisonedFL-ish local training (multi-round consistency + proximal)
# ----------------------------
def _param_l2(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = torch.tensor(0.0, device=CFG.device)
    for k in sd_a.keys():
        s = s + (sd_a[k] - sd_b[k]).pow(2).sum()
    return s


def local_train_honest(model: nn.Module, dataset, epochs: int) -> nn.Module:
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    opt = optim.SGD(model.parameters(), lr=CFG.lr, momentum=CFG.momentum, weight_decay=CFG.weight_decay)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(CFG.device), y.to(CFG.device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def local_train_poisonedfl(
    model: nn.Module,
    dataset,
    global_sd: Dict[str, torch.Tensor],
    prev_mal_tx_sd: Optional[Dict[str, torch.Tensor]],
    epochs: int,
) -> Dict[str, torch.Tensor]:
    """
    Lightweight-but-closer PoisonedFL-style:
      loss = CE(on modified labels) +
             mu * ||w - w_global||^2 +
             beta * ||w - w_prev_mal||^2
    """
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    opt = optim.SGD(model.parameters(), lr=CFG.lr, momentum=CFG.momentum, weight_decay=CFG.weight_decay)
    crit = nn.CrossEntropyLoss()

    head_arr = torch.tensor(list(CFG.head_classes), device=CFG.device, dtype=torch.long)
    tail_set = set(int(x) for x in CFG.tail_classes)

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(CFG.device), y.to(CFG.device)

            # degrade tail via flipping tail labels into head labels (cheap & consistent)
            y_mod = y
            if CFG.poisonedfl_tail_label_flip:
                y_mod = y.clone()
                tail_mask = torch.zeros_like(y_mod, dtype=torch.bool)
                for c in tail_set:
                    tail_mask |= (y_mod == c)
                if tail_mask.any() and len(head_arr) > 0:
                    ridx = torch.randint(0, len(head_arr), (tail_mask.sum().item(),), device=CFG.device)
                    y_mod[tail_mask] = head_arr[ridx]

            logits = model(x)
            loss_ce = crit(logits, y_mod)

            cur_sd = {k: v for k, v in model.state_dict().items()}
            loss_glob = _param_l2(cur_sd, global_sd)

            if prev_mal_tx_sd is not None:
                loss_cons = _param_l2(cur_sd, prev_mal_tx_sd)
            else:
                loss_cons = torch.tensor(0.0, device=CFG.device)

            loss = loss_ce + CFG.poisonedfl_mu_global * loss_glob + CFG.poisonedfl_beta_consistency * loss_cons

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model.state_dict()


# ----------------------------
# 7) Defenses: FedCC (linear CKA) + Attack-Adaptive Aggregation
# ----------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    if X.numel() == 0 or Y.numel() == 0:
        return 0.0
    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]

    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    XT_Y = Xc.t() @ Yc
    XT_X = Xc.t() @ Xc
    YT_Y = Yc.t() @ Yc

    num = (XT_Y.pow(2)).sum()
    den = torch.sqrt((XT_X.pow(2)).sum() * (YT_Y.pow(2)).sum() + eps)
    return float((num / (den + eps)).clamp(0.0, 1.0).item())


def defense_fedcc_cka_filter(
    num_classes: int,
    global_sd: Dict[str, torch.Tensor],
    client_tx_sds: List[Dict[str, torch.Tensor]],
    support_loader: DataLoader,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    g_model = get_model(num_classes)
    g_model.load_state_dict(global_sd)
    G = collect_features(g_model, support_loader)

    sims = []
    for tx_sd in client_tx_sds:
        c_model = get_model(num_classes)
        c_model.load_state_dict(tx_sd)
        C = collect_features(c_model, support_loader)
        sims.append(linear_cka(G, C))

    n = len(client_tx_sds)
    reject_n = int(round(CFG.fedcc_reject_frac * n))
    reject_n = min(reject_n, max(0, n - CFG.fedcc_min_keep))

    order = np.argsort(sims)  # ascending
    keep_idx = order[reject_n:].tolist()
    kept_sds = [client_tx_sds[i] for i in keep_idx]

    new_sd = fedavg_mean(kept_sds)

    info = {
        "defense": "FedCC_CKA",
        "cka_sim": sims,
        "keep_idx": keep_idx,
        "reject_idx": [i for i in range(n) if i not in set(keep_idx)],
    }
    return new_sd, info


def defense_attack_adaptive_agg(
    global_sd: Dict[str, torch.Tensor],
    client_tx_sds: List[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    updates = [flatten_update(global_sd, tx) for tx in client_tx_sds]
    U = torch.stack(updates, dim=0)

    med = U.median(dim=0).values
    med_norm = med.norm() + 1e-12

    norms = torch.tensor([u.norm().item() for u in updates], device=U.device)
    z = (norms - norms.mean()) / (norms.std() + 1e-12)

    cos_terms = []
    for u in updates:
        cos_terms.append((u @ med) / (u.norm() * med_norm + 1e-12))
    cos_terms = torch.stack(cos_terms, dim=0).clamp(-1.0, 1.0)
    cos_dist = 1.0 - cos_terms

    score = z + cos_dist
    weights = torch.softmax(-CFG.adaagg_alpha * score, dim=0).detach().cpu().numpy().tolist()

    new_sd = fedavg_mean(client_tx_sds, weights=weights)

    info = {
        "defense": "AttackAdaptiveAggregation",
        "scores": score.detach().cpu().numpy().tolist(),
        "weights": weights,
    }
    return new_sd, info


# ----------------------------
# 8) Run one FL (attack x defense)
# ----------------------------
def tail_slice(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) <= k:
        return x
    return x[-k:]


def mean_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def std_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.std(xs)) if len(xs) > 0 else float("nan")


def run_fl_once(
    num_classes: int,
    client_train_sets: List[TensorDataset],
    client_test_sets: List[TensorDataset],
    public_support: TensorDataset,
    public_head_eval: TensorDataset,
    public_tail_eval: TensorDataset,
    defense_name: str,  # "FedCC" | "AttackAdaptiveAggregation"
    attack_name: str,   # "NONE" | "POISONEDFL" | "BACKDOOR"
    seed_offset: int = 0,
) -> Dict:
    rng = np.random.default_rng(CFG.seed + seed_offset)

    n_clients = len(client_train_sets)
    n_mal = int(round(CFG.malicious_frac * n_clients))
    all_ids = np.arange(n_clients)
    rng.shuffle(all_ids)
    malicious_ids = set(all_ids[:n_mal].tolist())

    # build per-client train datasets (wrap backdoor for malicious)
    built_train_sets = []
    for cid in range(n_clients):
        base = client_train_sets[cid]
        if cid in malicious_ids and attack_name == "BACKDOOR":
            built = BackdoorWrapper(
                base=base,
                poison_frac=CFG.backdoor_poison_frac,
                target_label=CFG.backdoor_target_label,
                trigger_size=CFG.trigger_size,
                trigger_value=CFG.trigger_value,
                seed=int(CFG.seed + 777 + seed_offset + cid),
            )
            built_train_sets.append(built)
        else:
            built_train_sets.append(base)

    # loaders
    support_loader = DataLoader(public_support, batch_size=CFG.batch_size, shuffle=False)
    head_eval_loader = DataLoader(public_head_eval, batch_size=CFG.batch_size, shuffle=False)
    tail_eval_loader = DataLoader(public_tail_eval, batch_size=CFG.batch_size, shuffle=False)

    # welfare: tail accuracy across clients on their test-tail subsets
    client_tail_tests = [filter_by_labels(ds, CFG.tail_classes) for ds in client_test_sets]

    # init global
    global_model = get_model(num_classes)
    global_sd = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    # PoisonedFL memory
    prev_mal_tx_sd: Dict[int, Optional[Dict[str, torch.Tensor]]] = {cid: None for cid in malicious_ids}

    last_reject_rate = 0.0

    # histories
    M_head_hist, M_tail_hist, W_tail_hist, gap_hist = [], [], [], []
    reject_rate_hist = []
    scale_hist = []

    for rnd in range(CFG.n_rounds):
        part = rng.choice(n_clients, size=min(CFG.clients_per_round, n_clients), replace=False).tolist()
        client_tx_sds = []

        for cid in part:
            local_model = get_model(num_classes)
            local_model.load_state_dict(global_sd)

            is_attack_active = (attack_name != "NONE") and (rnd >= CFG.attack_start_round) and (cid in malicious_ids)

            if is_attack_active and attack_name == "POISONEDFL":
                local_sd = local_train_poisonedfl(
                    model=local_model,
                    dataset=built_train_sets[cid],
                    global_sd=global_sd,
                    prev_mal_tx_sd=prev_mal_tx_sd[cid],
                    epochs=CFG.local_epochs,
                )

                # dynamic magnitude adjustment based on last reject rate (FedCC only; else 0)
                scale = CFG.poisonedfl_scale * (1.0 + CFG.poisonedfl_dyn_eta * (last_reject_rate - 0.25))
                scale = float(np.clip(scale, CFG.poisonedfl_scale_min, CFG.poisonedfl_scale_max))
                scale_hist.append(scale)

                tx_sd = make_transmitted_state(global_sd, local_sd, scale=scale)
                client_tx_sds.append(tx_sd)

                prev_mal_tx_sd[cid] = {k: v.detach().clone() for k, v in tx_sd.items()}

            else:
                # honest train (or backdoor already injected into data)
                local_model = local_train_honest(local_model, built_train_sets[cid], epochs=CFG.local_epochs)
                local_sd = local_model.state_dict()

                if is_attack_active and attack_name == "BACKDOOR":
                    scale = CFG.model_replacement_gamma
                else:
                    scale = 1.0
                scale_hist.append(scale)

                tx_sd = make_transmitted_state(global_sd, local_sd, scale=scale)
                client_tx_sds.append(tx_sd)

        # defense
        if defense_name == "FedCC":
            new_sd, info = defense_fedcc_cka_filter(
                num_classes=num_classes,
                global_sd=global_sd,
                client_tx_sds=client_tx_sds,
                support_loader=support_loader,
            )
            rej = len(info["reject_idx"])
            last_reject_rate = float(rej / max(1, len(client_tx_sds)))
        elif defense_name == "AttackAdaptiveAggregation":
            new_sd, info = defense_attack_adaptive_agg(global_sd=global_sd, client_tx_sds=client_tx_sds)
            last_reject_rate = 0.0
        else:
            raise ValueError(f"Unknown defense: {defense_name}")

        global_sd = new_sd
        global_model.load_state_dict(global_sd)

        # public head/tail metric
        M_head = evaluate_acc(global_model, head_eval_loader)
        M_tail = evaluate_acc(global_model, tail_eval_loader)

        # welfare: mean tail acc across clients
        tail_accs = []
        for ds_tail in client_tail_tests:
            if len(ds_tail) == 0:
                continue
            loader = DataLoader(ds_tail, batch_size=CFG.batch_size, shuffle=False)
            tail_accs.append(evaluate_acc(global_model, loader))
        W_tail = float(np.mean(tail_accs)) if len(tail_accs) > 0 else 0.0

        gap = float(M_head - W_tail)

        M_head_hist.append(M_head)
        M_tail_hist.append(M_tail)
        W_tail_hist.append(W_tail)
        gap_hist.append(gap)
        reject_rate_hist.append(last_reject_rate)

        if (rnd + 1) % 10 == 0 or rnd == 0:
            print(
                f"[{attack_name} | {defense_name}] "
                f"Round {rnd+1:>3}/{CFG.n_rounds} | "
                f"M_head={M_head:.4f} M_tail={M_tail:.4f} W_tail={W_tail:.4f} gap={gap:.4f} | "
                f"rej_rate={last_reject_rate:.2f}"
            )

    return {
        "attack": attack_name,
        "defense": defense_name,
        "n_clients": n_clients,
        "clients_per_round": CFG.clients_per_round,
        "malicious_frac": CFG.malicious_frac,
        "n_malicious": n_mal,
        "attack_start_round": CFG.attack_start_round,
        "M_head_hist": np.array(M_head_hist, dtype=np.float32),
        "M_tail_hist": np.array(M_tail_hist, dtype=np.float32),
        "W_tail_hist": np.array(W_tail_hist, dtype=np.float32),
        "gap_hist": np.array(gap_hist, dtype=np.float32),
        "reject_rate_hist": np.array(reject_rate_hist, dtype=np.float32),
        "scale_hist": np.array(scale_hist, dtype=np.float32),
    }


# ----------------------------
# 9) Threshold + summary metrics
# ----------------------------
def compute_alarm_threshold_from_baseline(gap_hist: np.ndarray) -> float:
    xs = tail_slice(gap_hist, CFG.tail_k).astype(np.float64)
    mu = float(np.mean(xs))
    sd = float(np.std(xs))
    return mu + CFG.alarm_threshold_std * sd


def fp_fn_ewdelay(gap_hist: np.ndarray, thr: float, attack_start_round: int, is_attack: bool) -> Tuple[float, float, float]:
    T = len(gap_hist)
    g = gap_hist.astype(np.float64)

    if not is_attack:
        start = min(T, CFG.warmup_ignore)
        fp = float(np.mean(g[start:] > thr)) if T > start else 0.0
        return fp, float("nan"), float("nan")

    start = min(T, attack_start_round)
    post = g[start:]
    fn = float(np.mean(post <= thr)) if len(post) > 0 else float("nan")
    ew = float("nan")
    for t in range(start, T):
        if g[t] > thr:
            ew = float(t - start)
            break
    return float("nan"), fn, ew


def summarize_run(run: Dict, thr: float, baseline_W: Optional[float]) -> Dict:
    M_head = mean_last_k(run["M_head_hist"], CFG.tail_k)
    M_tail = mean_last_k(run["M_tail_hist"], CFG.tail_k)
    W_tail = mean_last_k(run["W_tail_hist"], CFG.tail_k)
    gap = mean_last_k(run["gap_hist"], CFG.tail_k)

    pog = float("nan")
    if baseline_W is not None and np.isfinite(baseline_W) and baseline_W > 1e-8:
        pog = float((baseline_W - W_tail) / baseline_W)

    is_attack = (run["attack"] != "NONE")
    fp, fn, ew = fp_fn_ewdelay(run["gap_hist"], thr, CFG.attack_start_round, is_attack=is_attack)

    return {
        "condition": "ATTACK" if is_attack else "BASELINE",
        "attack": run["attack"],
        "defense": run["defense"],
        "n_clients": run["n_clients"],
        "clients_per_round": run["clients_per_round"],
        "malicious_frac": run["malicious_frac"],
        "n_malicious": run["n_malicious"],
        "attack_start_round": run["attack_start_round"],
        "threshold_gap": float(thr),

        "M_head_lastK": float(M_head),
        "M_tail_lastK": float(M_tail),
        "W_tail_lastK": float(W_tail),
        "gap_Mhead_minus_W_lastK": float(gap),

        "PoG_vs_baselineW_same_defense": float(pog),
        "FP_rate_baseline": float(fp),
        "FN_rate_attack": float(fn),
        "EW_delay_rounds": float(ew),

        "gap_std_lastK": float(std_last_k(run["gap_hist"], CFG.tail_k)),
        "W_std_lastK": float(std_last_k(run["W_tail_hist"], CFG.tail_k)),

        "avg_reject_rate": float(np.mean(run["reject_rate_hist"])) if len(run["reject_rate_hist"]) > 0 else float("nan"),
    }


# ----------------------------
# 10) Main: baselines per defense + 2x2 runs + CSV
# ----------------------------
def main():
    print("\n=== E4 (No LEAF): FEMNIST | 2x2 Attacks x Defenses ===")
    print(f"Device: {CFG.device}")
    print(f"n_clients={CFG.n_clients}, clients_per_round={CFG.clients_per_round}, rounds={CFG.n_rounds}, local_epochs={CFG.local_epochs}")
    print(f"malicious_frac={CFG.malicious_frac}, attack_start_round={CFG.attack_start_round}")
    print(f"public_support_size={CFG.public_support_size} (CKA)\n")

    rng = np.random.default_rng(CFG.seed)

    # 1) Load FEMNIST clients (writer_id = client) without LEAF
    client_train_sets, client_test_sets, num_classes = build_femnist_clients_from_hf(
        n_clients=CFG.n_clients,
        train_ratio=0.8,
        seed=CFG.seed,
        min_train=CFG.min_train_samples_per_client,
        min_test=CFG.min_test_samples_per_client,
        max_try=CFG.max_partition_id_try,
    )
    print(f"Collected clients: {len(client_train_sets)}")
    print(f"Inferred num_classes: {num_classes}")

    # 2) Public sets from client train data
    public_head = sample_public_from_clients(client_train_sets, CFG.head_classes, CFG.public_head_eval_size, rng)
    public_tail = sample_public_from_clients(client_train_sets, CFG.tail_classes, CFG.public_tail_eval_size, rng)
    public_support = sample_public_from_clients(client_train_sets, CFG.head_classes, CFG.public_support_size, rng)
    if len(public_support) == 0:
        public_support = sample_public_from_clients(client_train_sets, tuple(range(num_classes)), CFG.public_support_size, rng)

    if len(public_head) == 0 or len(public_tail) == 0:
        raise RuntimeError(
            "Public head/tail eval sets are empty. "
            "Adjust head_classes/tail_classes or increase public_*_eval_size."
        )

    defenses = ["FedCC", "AttackAdaptiveAggregation"]
    attacks = ["POISONEDFL", "BACKDOOR"]

    seed_offset = 0
    baseline_cache: Dict[str, Tuple[float, float]] = {}  # defense -> (thr, baseline_W)
    rows: List[Dict] = []

    # 3) Baselines per defense (attack=NONE), used to set alarm threshold per defense
    for defense in defenses:
        print(f"\n--- BASELINE (NONE) | defense={defense} ---")
        base = run_fl_once(
            num_classes=num_classes,
            client_train_sets=client_train_sets,
            client_test_sets=client_test_sets,
            public_support=public_support,
            public_head_eval=public_head,
            public_tail_eval=public_tail,
            defense_name=defense,
            attack_name="NONE",
            seed_offset=seed_offset,
        )
        seed_offset += 1

        thr = compute_alarm_threshold_from_baseline(base["gap_hist"])
        Wb = mean_last_k(base["W_tail_hist"], CFG.tail_k)
        baseline_cache[defense] = (thr, Wb)

        row = summarize_run(base, thr=thr, baseline_W=None)
        rows.append(row)

        np.savez(os.path.join(CFG.out_dir, f"E4_hist_BASELINE_{defense}.npz"), **base)

    # 4) 2x2 runs
    for attack in attacks:
        for defense in defenses:
            print(f"\n=== RUN | attack={attack} | defense={defense} ===")
            thr, Wb = baseline_cache[defense]

            out = run_fl_once(
                num_classes=num_classes,
                client_train_sets=client_train_sets,
                client_test_sets=client_test_sets,
                public_support=public_support,
                public_head_eval=public_head,
                public_tail_eval=public_tail,
                defense_name=defense,
                attack_name=attack,
                seed_offset=seed_offset,
            )
            seed_offset += 1

            row = summarize_run(out, thr=thr, baseline_W=Wb)
            rows.append(row)

            np.savez(os.path.join(CFG.out_dir, f"E4_hist_{attack}_{defense}.npz"), **out)

    # 5) Save CSV
    csv_path = os.path.join(CFG.out_dir, CFG.csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved histories (.npz) in: {CFG.out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()