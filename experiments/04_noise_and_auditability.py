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
# E2: Noise / Privacy Trade-off (DP-like update noise injection)
# Standalone code (separate from your E1 script)
#
# Changes requested:
# - noise_multipliers = (0.00, 0.05, 0.10)
# - keep the rest of settings the same as the previous E2 code
# - add more measured metrics
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

    # Public leak (for GAMING_HEAD_ONLY_LEAK)
    leak_fraction: float = 1.0

    # Strategy mix
    benign_frac: float = 0.10
    gaming_frac_main: float = 0.30  # E2's "gaming" condition
    update_scale_factor: float = 1.75  # for update-scaling gaming

    # DP-like noise injection (privacy knob)
    # noise_multiplier = sigma, noise std = sigma * clip_C
    clip_C: float = 1.0
    noise_multipliers: Tuple[float, ...] = (0.00, 0.05, 0.10)  # <-- requested

    # Evaluation summary
    tail_k: int = 7

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
# - 50k for client-local data
# - 10k as public validation candidates (from train)
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
    for idx_in_local in base_subset.indices:
        global_idx = train_local_set.indices[idx_in_local]  # index into full_train
        y = int(full_train.targets[global_idx])
        if y in allowed:
            kept.append(idx_in_local)
    return Subset(train_local_set, kept)


def split_subset_train_eval(base_subset: Subset, eval_ratio: float = 0.2) -> Tuple[Subset, Subset]:
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

client_train_subsets: List[Subset] = []
client_eval_subsets: List[Subset] = []
client_tail_eval_subsets: List[Subset] = []
client_tail_eval_sizes: List[int] = []

for cid in range(CFG.n_clients):
    tr, ev = split_subset_train_eval(client_base_subsets[cid], eval_ratio=0.2)
    client_train_subsets.append(tr)
    client_eval_subsets.append(ev)
    tail_ev = tail_only_subset(ev)
    client_tail_eval_subsets.append(tail_ev)
    client_tail_eval_sizes.append(len(tail_ev))


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


def fedavg_from_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
        return oversample_tail(base_train, factor=2)

    if strategy == STR_GAME_HEAD:
        head_only = filter_subset_by_labels(base_train, CFG.head_classes)
        return ConcatDataset([head_only, leak_subset])

    if strategy == STR_GAME_SCALE:
        return base_train

    raise ValueError(f"Unknown strategy: {strategy}")


# ----------------------------
# 6) DP-like noise injection on transmitted update
# ----------------------------
def _flatten_delta(delta_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    vecs = []
    for k in delta_sd.keys():
        vecs.append(delta_sd[k].reshape(-1))
    return torch.cat(vecs, dim=0)


def _unflatten_to_sd(vec: torch.Tensor, template_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    offset = 0
    for k in template_sd.keys():
        numel = template_sd[k].numel()
        out[k] = vec[offset:offset + numel].reshape(template_sd[k].shape)
        offset += numel
    return out


def dp_perturb_delta(delta_sd: Dict[str, torch.Tensor], clip_C: float, noise_mult: float) -> Dict[str, torch.Tensor]:
    """
    DP-SGD-like: clip global L2 norm of the update delta, then add Gaussian noise.
    noise std = noise_mult * clip_C
    """
    delta_vec = _flatten_delta(delta_sd)

    # Clip
    norm = torch.norm(delta_vec, p=2)
    if norm > clip_C:
        delta_vec = delta_vec * (clip_C / (norm + 1e-12))

    # Noise
    if noise_mult > 0:
        std = noise_mult * clip_C
        delta_vec = delta_vec + torch.randn_like(delta_vec) * std

    return _unflatten_to_sd(delta_vec, delta_sd)


def make_transmitted_state(global_sd: Dict[str, torch.Tensor],
                           local_sd: Dict[str, torch.Tensor],
                           scale: float,
                           clip_C: float,
                           noise_mult: float) -> Dict[str, torch.Tensor]:
    """
    transmitted = global + DP( scale * (local - global) )
    """
    delta = {}
    for k in global_sd.keys():
        delta[k] = scale * (local_sd[k] - global_sd[k])

    delta = dp_perturb_delta(delta, clip_C=clip_C, noise_mult=noise_mult)

    tx = {}
    for k in global_sd.keys():
        tx[k] = global_sd[k] + delta[k]
    return tx


# ----------------------------
# 7) Run FL once (given gaming_frac and noise_mult)
#    Per-round:
#    - M_t: public head accuracy (metric)
#    - W_t: mean tail accuracy across clients (welfare)
#    - W_std_t: std tail acc across clients
#    - W_min_t, W_max_t: min/max tail acc across clients
#    - tail_nonempty_count_t: number of clients with tail eval > 0
# ----------------------------
def run_once(gaming_frac: float, benign_frac: float, noise_mult: float, seed_offset: int = 0):
    rng = np.random.default_rng(CFG.seed + seed_offset)
    strat_map = assign_strategies(CFG.n_clients, gaming_frac, benign_frac, rng)

    client_train_ds = [build_client_train_dataset(cid, strat_map[cid]) for cid in range(CFG.n_clients)]

    global_model = get_model()
    global_sd = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    M_hist: List[float] = []
    W_hist: List[float] = []
    W_std_hist: List[float] = []
    W_min_hist: List[float] = []
    W_max_hist: List[float] = []
    tail_nonempty_hist: List[int] = []

    # For reporting how many are gaming/benign/honest (fixed for the run)
    n_honest = sum(1 for s in strat_map.values() if s == STR_HONEST)
    n_benign = sum(1 for s in strat_map.values() if s == STR_BENIGN)
    n_ghead = sum(1 for s in strat_map.values() if s == STR_GAME_HEAD)
    n_gscale = sum(1 for s in strat_map.values() if s == STR_GAME_SCALE)

    for rnd in range(CFG.n_rounds):
        transmitted_states: List[Dict[str, torch.Tensor]] = []

        for cid in range(CFG.n_clients):
            local_model = get_model()
            local_model.load_state_dict(global_sd)

            local_model = local_train(local_model, client_train_ds[cid], epochs=CFG.local_epochs)
            local_sd = local_model.state_dict()

            # scale for UPDATE_SCALING gaming; others scale=1
            if strat_map[cid] == STR_GAME_SCALE and gaming_frac > 0:
                scale = CFG.update_scale_factor
            else:
                scale = 1.0

            # DP-like perturbation on transmitted update (applies to everyone)
            tx_sd = make_transmitted_state(
                global_sd=global_sd,
                local_sd=local_sd,
                scale=scale,
                clip_C=CFG.clip_C,
                noise_mult=noise_mult
            )
            transmitted_states.append(tx_sd)

        # FedAvg update
        global_sd = fedavg_from_states(transmitted_states)
        global_model.load_state_dict(global_sd)

        # Metric: server-observable public head validation accuracy
        M_t = evaluate_acc(global_model, public_val_loader)
        M_hist.append(M_t)

        # Welfare: tail accuracy distribution across clients (experimenter-side evaluation)
        tail_accs = []
        nonempty = 0
        for cid in range(CFG.n_clients):
            tail_ev = client_tail_eval_subsets[cid]
            if len(tail_ev) == 0:
                continue
            nonempty += 1
            loader = DataLoader(tail_ev, batch_size=CFG.batch_size, shuffle=False)
            tail_accs.append(evaluate_acc(global_model, loader))

        tail_nonempty_hist.append(nonempty)

        if len(tail_accs) > 0:
            W_t = float(np.mean(tail_accs))
            W_std_t = float(np.std(tail_accs))
            W_min_t = float(np.min(tail_accs))
            W_max_t = float(np.max(tail_accs))
        else:
            W_t, W_std_t, W_min_t, W_max_t = 0.0, float("nan"), float("nan"), float("nan")

        W_hist.append(W_t)
        W_std_hist.append(W_std_t)
        W_min_hist.append(W_min_t)
        W_max_hist.append(W_max_t)

        if (rnd + 1) % 10 == 0 or rnd == 0:
            print(f"[gf={gaming_frac:.2f} | noise={noise_mult:.2f}] "
                  f"Round {rnd+1:>2}/{CFG.n_rounds} | M(head)={M_t:.4f} | "
                  f"W_tail(mean)={W_t:.4f} | W_std={W_std_t:.4f} | n_tail_clients={nonempty}")

    return {
        "gaming_frac": gaming_frac,
        "benign_frac": benign_frac,
        "noise_mult": noise_mult,
        "strategy_map": strat_map,
        "strategy_counts": {
            "honest": n_honest,
            "benign": n_benign,
            "gaming_head": n_ghead,
            "gaming_scale": n_gscale,
        },
        "M_hist": np.array(M_hist, dtype=np.float32),
        "W_hist": np.array(W_hist, dtype=np.float32),
        "W_std_hist": np.array(W_std_hist, dtype=np.float32),
        "W_min_hist": np.array(W_min_hist, dtype=np.float32),
        "W_max_hist": np.array(W_max_hist, dtype=np.float32),
        "tail_nonempty_hist": np.array(tail_nonempty_hist, dtype=np.int32),
    }


# ----------------------------
# 8) Summary metrics
# ----------------------------
def tail_slice(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return x
    if len(x) <= k:
        return x
    return x[-k:]


def mean_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def std_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.std(xs)) if len(xs) > 0 else float("nan")


def min_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.min(xs)) if len(xs) > 0 else float("nan")


def max_last_k(x: np.ndarray, k: int) -> float:
    xs = tail_slice(x, k)
    return float(np.max(xs)) if len(xs) > 0 else float("nan")


def compute_pog(W_ref: float, W_final: float) -> float:
    if not np.isfinite(W_ref) or W_ref <= 1e-8:
        return float("nan")
    return float((W_ref - W_final) / W_ref)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))


# ----------------------------
# 9) Main: Sweep noise levels, compare aligned vs gaming
# ----------------------------
def main():
    print("\n=== E2: Noise / Privacy Trade-off (DP-like update noise) ===")
    print(f"Device: {CFG.device}")
    print(f"Clients={CFG.n_clients}, Rounds={CFG.n_rounds}, LocalEpochs={CFG.local_epochs}")
    print(f"clip_C={CFG.clip_C}, noise_multipliers={CFG.noise_multipliers}")
    print(f"benign_frac={CFG.benign_frac}, gaming_frac_main={CFG.gaming_frac_main}\n")

    # 9.1 Reference baseline (aligned, noise=0)
    print(">> Running reference baseline: gf=0.0, noise=0.0")
    ref = run_once(gaming_frac=0.0, benign_frac=CFG.benign_frac, noise_mult=0.0, seed_offset=0)
    M_ref = mean_last_k(ref["M_hist"], CFG.tail_k)
    W_ref = mean_last_k(ref["W_hist"], CFG.tail_k)

    print("\n[Reference] tail-mean over last K rounds")
    print(f"  M_ref (head metric) = {M_ref:.4f}")
    print(f"  W_ref (tail welfare)= {W_ref:.4f}\n")

    # 9.2 Sweep noise for two conditions: aligned (gf=0) vs gaming (gf=gaming_frac_main)
    results = []
    seed_offset = 10
    for nm in CFG.noise_multipliers:
        # aligned under noise
        print(f"\n>> Running ALIGNED under noise={nm:.2f} (gf=0.0)")
        aligned = run_once(gaming_frac=0.0, benign_frac=CFG.benign_frac, noise_mult=nm, seed_offset=seed_offset)
        seed_offset += 1

        # gaming under noise
        print(f"\n>> Running GAMING under noise={nm:.2f} (gf={CFG.gaming_frac_main:.2f})")
        gaming = run_once(gaming_frac=CFG.gaming_frac_main, benign_frac=CFG.benign_frac, noise_mult=nm, seed_offset=seed_offset)
        seed_offset += 1

        # Compute summary metrics for each condition
        def summarize(tag: str, out: Dict):
            M_final = mean_last_k(out["M_hist"], CFG.tail_k)
            W_final = mean_last_k(out["W_hist"], CFG.tail_k)
            gap = M_final - W_final

            # "Classic" PoG vs reference W_ref (kept as-is; does NOT change your paper's definition)
            pog = compute_pog(W_ref, W_final)

            # Additional metrics (more stable / more informative under noise)
            M_std = std_last_k(out["M_hist"], CFG.tail_k)
            W_std = std_last_k(out["W_hist"], CFG.tail_k)

            # Tail distribution stats across clients (per-round std/min/max, summarized over last K)
            W_std_across_clients_mean = mean_last_k(out["W_std_hist"], CFG.tail_k)
            W_min = mean_last_k(out["W_min_hist"], CFG.tail_k)
            W_max = mean_last_k(out["W_max_hist"], CFG.tail_k)

            # Stability / spread
            range_M = max_last_k(out["M_hist"], CFG.tail_k) - min_last_k(out["M_hist"], CFG.tail_k)
            range_W = max_last_k(out["W_hist"], CFG.tail_k) - min_last_k(out["W_hist"], CFG.tail_k)

            # Deltas vs reference
            dM_vs_ref = M_final - M_ref
            dW_vs_ref = W_final - W_ref

            # Normalized gap ratio (how large is gap relative to welfare)
            gap_ratio = safe_div(gap, max(W_final, 0.0))

            # Average number of clients with non-empty tail eval (should be constant, but log anyway)
            tail_nonempty = int(np.round(mean_last_k(out["tail_nonempty_hist"].astype(np.float32), CFG.tail_k)))

            sc = out["strategy_counts"]

            return {
                "condition": tag,
                "noise_mult": float(out["noise_mult"]),
                "gaming_frac": float(out["gaming_frac"]),
                "benign_frac": float(out["benign_frac"]),
                "n_clients": int(CFG.n_clients),
                "n_honest": int(sc["honest"]),
                "n_benign": int(sc["benign"]),
                "n_gaming_head": int(sc["gaming_head"]),
                "n_gaming_scale": int(sc["gaming_scale"]),

                # Primary end metrics
                "M_final": float(M_final),
                "W_tail_final": float(W_final),
                "gap_M_minus_W": float(gap),
                "gap_ratio_gap_over_W": float(gap_ratio),
                "PoG_vs_ref": float(pog),

                # Deltas vs reference
                "M_delta_vs_ref": float(dM_vs_ref),
                "W_delta_vs_ref": float(dW_vs_ref),

                # Within-run temporal stability (last K rounds)
                "M_std_lastK": float(M_std),
                "W_std_lastK": float(W_std),
                "M_range_lastK": float(range_M),
                "W_range_lastK": float(range_W),

                # Across-client tail distribution stats (last K rounds averaged)
                "W_std_across_clients_lastK": float(W_std_across_clients_mean),
                "W_min_across_clients_lastK": float(W_min),
                "W_max_across_clients_lastK": float(W_max),

                # Tail eval coverage
                "tail_nonempty_clients_lastK": int(tail_nonempty),
            }

        row_aligned = summarize("ALIGNED", aligned)
        row_gaming = summarize("GAMING", gaming)

        # Paired "extra effect" metrics at the same noise (does NOT change PoG definition)
        # These help you talk about "additional harm under gaming" without redefining PoG.
        deltaW_same_noise = row_aligned["W_tail_final"] - row_gaming["W_tail_final"]
        deltaGap_same_noise = row_gaming["gap_M_minus_W"] - row_aligned["gap_M_minus_W"]

        # Relative additional harm ratio (if aligned welfare is too small, it will blow up; keep it but be cautious)
        rel_deltaW = safe_div(deltaW_same_noise, max(row_aligned["W_tail_final"], 0.0))

        # Store rows with paired deltas
        row_aligned["paired_deltaW_aligned_minus_gaming"] = float(deltaW_same_noise)
        row_aligned["paired_deltaGap_gaming_minus_aligned"] = float(deltaGap_same_noise)
        row_aligned["paired_rel_deltaW_over_Waligned"] = float(rel_deltaW)

        row_gaming["paired_deltaW_aligned_minus_gaming"] = float(deltaW_same_noise)
        row_gaming["paired_deltaGap_gaming_minus_aligned"] = float(deltaGap_same_noise)
        row_gaming["paired_rel_deltaW_over_Waligned"] = float(rel_deltaW)

        results.append(row_aligned)
        results.append(row_gaming)

        # quick print
        print(f"\n[Summary @ noise={nm:.2f}]")
        print(f"  ALIGNED: M={row_aligned['M_final']:.4f} | W={row_aligned['W_tail_final']:.4f} | "
              f"gap={row_aligned['gap_M_minus_W']:.4f} | PoG(ref)={row_aligned['PoG_vs_ref']:.4f}")
        print(f"  GAMING : M={row_gaming['M_final']:.4f} | W={row_gaming['W_tail_final']:.4f} | "
              f"gap={row_gaming['gap_M_minus_W']:.4f} | PoG(ref)={row_gaming['PoG_vs_ref']:.4f}")
        print(f"  Paired: ΔW(aligned-gaming)={deltaW_same_noise:+.4f} | "
              f"Δgap(gaming-aligned)={deltaGap_same_noise:+.4f} | relΔW={rel_deltaW:+.4f}")

    # 9.3 Save CSV
    out_path = "E2_noise_privacy_tradeoff_results.csv"
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()