import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

# ============================================================
# 0. Config / Seed
# ============================================================

class Config:
    n_clients = 30
    gaming_frac = 0.3
    n_rounds = 40
    local_epochs = 2
    batch_size = 64
    lr = 0.01
    momentum = 0.9

    # Dirichlet non-IID partition parameter
    dirichlet_alpha = 0.5

    # Fraction of public validation data leaked to gaming clients
    leak_fraction = 1.0  # use entire head-only public val for maximum effect

    # Class split
    head_classes = {0, 1, 2, 3, 4}   # head classes used by the public metric
    tail_classes = {5, 6, 7, 8, 9}   # tail classes used by welfare

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(Config.seed)

# ============================================================
# 1. Dataset preparation (Fashion-MNIST)
#    - train 60k -> train_local 50k + public_val (head-biased) ~10k
#    - test 10k  -> hidden welfare evaluation on tail-only
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

root = "./data"

full_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

# From the 60k training examples:
#   first 50k: local training
#   last 10k : public validation candidates
train_local_size = 50_000
indices_all = np.arange(len(full_train))
train_local_indices = indices_all[:train_local_size]
public_val_candidate_indices = indices_all[train_local_size:]

train_local_set = Subset(full_train, train_local_indices)

# Public validation: only head classes
all_train_targets = np.array(full_train.targets)
head_mask = np.isin(all_train_targets, list(Config.head_classes))
public_val_head_indices = [idx for idx in public_val_candidate_indices if head_mask[idx]]

public_val_head_indices = np.array(public_val_head_indices)
rng = np.random.default_rng(Config.seed)
rng.shuffle(public_val_head_indices)

leak_size = int(Config.leak_fraction * len(public_val_head_indices))
leak_indices = public_val_head_indices[:leak_size]

public_val_set = Subset(full_train, public_val_head_indices)
public_val_loader = DataLoader(public_val_set, batch_size=Config.batch_size, shuffle=False)

# Hidden welfare: only tail classes from the test set
all_test_targets = np.array(test_set.targets)
tail_mask = np.isin(all_test_targets, list(Config.tail_classes))
tail_test_indices = np.where(tail_mask)[0]
hidden_tail_test_set = Subset(test_set, tail_test_indices)
hidden_tail_test_loader = DataLoader(hidden_tail_test_set, batch_size=Config.batch_size, shuffle=False)

# (Optional) If we want full test accuracy later
full_test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)

# Public leak subset for gaming clients (head-only)
leak_subset = Subset(full_train, leak_indices)


# ============================================================
# 2. Non-IID partitioning (Dirichlet) for client local datasets
# ============================================================

def make_dirichlet_partitions(labels: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator):
    """
    labels: 1D array of class labels for train_local_set
    n_clients: number of clients
    alpha: Dirichlet concentration parameter

    Returns:
        list of index arrays per client (indices are with respect to train_local_set)
    """
    n_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(n_clients)]

    # Collect indices per class (indices are within train_local_set)
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
        shard = np.split(idx_k, splits[:-1])

        for cid, shard_c in enumerate(shard):
            client_indices[cid].extend(shard_c.tolist())

    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


rng = np.random.default_rng(Config.seed)

# Extract labels for train_local_set (in the order of train_local_indices)
train_targets = np.array(full_train.targets)[train_local_indices]

client_indices = make_dirichlet_partitions(
    labels=train_targets,
    n_clients=Config.n_clients,
    alpha=Config.dirichlet_alpha,
    rng=rng
)

client_datasets_base = [
    Subset(train_local_set, idxs) for idxs in client_indices
]


# ============================================================
# 3. Helper to filter a Subset by allowed labels
# ============================================================

def filter_subset_by_labels(base_subset: Subset, allowed_labels: set[int]) -> Subset:
    """
    base_subset: Subset(train_local_set, ...)
    allowed_labels: set of class labels to keep (e.g., head_classes)

    Returns:
        A new Subset(train_local_set, ...) that only includes samples whose
        labels are in allowed_labels.
    """
    assert isinstance(base_subset.dataset, Subset) or isinstance(base_subset.dataset, datasets.FashionMNIST) \
        or isinstance(base_subset.dataset, torch.utils.data.Dataset)

    # base_subset is a subset of train_local_set.
    # train_local_set.indices: indices into full_train.
    # base_subset.indices: indices within train_local_set (0~49999).
    kept_indices_in_train_local = []

    for idx_in_train_local in base_subset.indices:
        global_idx = train_local_set.indices[idx_in_train_local]
        label = int(full_train.targets[global_idx])
        if label in allowed_labels:
            kept_indices_in_train_local.append(idx_in_train_local)

    return Subset(train_local_set, kept_indices_in_train_local)


# ============================================================
# 4. CNN model definition
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1x28x28 -> 32x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 64x7x7
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


def get_model():
    model = SimpleCNN(num_classes=10)
    return model.to(Config.device)


# ============================================================
# 5. Evaluation function (accuracy)
# ============================================================

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(Config.device)
            y = y.to(Config.device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ============================================================
# 6. Local training
#    (honest vs gaming is controlled by which dataset they receive)
# ============================================================

def local_train(model: nn.Module, dataset, epochs: int) -> nn.Module:
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr, momentum=Config.momentum)

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(Config.device)
            y = y.to(Config.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model


# ============================================================
# 7. FedAvg aggregation
# ============================================================

def fedavg(models: list[nn.Module]) -> nn.Module:
    global_model = get_model()
    global_state = global_model.state_dict()

    state_dicts = [m.state_dict() for m in models]

    with torch.no_grad():
        for key in global_state.keys():
            stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
            global_state[key] = stacked.mean(dim=0)

    global_model.load_state_dict(global_state)
    return global_model


# ============================================================
# 8. Federated learning experiment (only gaming_frac changes)
# ============================================================

def run_federated_experiment(gaming_frac: float):
    """
    gaming_frac: 0.0 => all honest (aligned)
                 0.3 => 30% gaming

    Returns:
      - W_history: per-round hidden tail test accuracy (welfare)
      - M_history: per-round head-only public validation accuracy (metric)
      - overall_test_history: per-round full test accuracy (for reference)
    """
    n_clients = Config.n_clients
    n_gaming = int(round(gaming_frac * n_clients))

    all_client_ids = np.arange(n_clients)
    rng_local = np.random.default_rng(Config.seed)  # fix seed to keep client composition consistent
    gaming_clients = set(rng_local.choice(all_client_ids, size=n_gaming, replace=False))
    honest_clients = [cid for cid in all_client_ids if cid not in gaming_clients]

    print(f"[Run] gaming_frac = {gaming_frac:.2f}, "
          f"honest = {len(honest_clients)}, gaming = {len(gaming_clients)}")

    # Construct training dataset for each client
    client_train_datasets = []
    for cid in range(n_clients):
        base_ds = client_datasets_base[cid]
        if cid in gaming_clients and gaming_frac > 0:
            # Gaming client:
            #   1) Remove tail classes from local data (head-only)
            #   2) Concatenate head-only public leak
            base_head_only = filter_subset_by_labels(base_ds, Config.head_classes)
            ds = ConcatDataset([base_head_only, leak_subset])
        else:
            # Honest client: use entire local data (head + tail)
            ds = base_ds
        client_train_datasets.append(ds)

    global_model = get_model()

    W_history = []
    M_history = []
    overall_test_history = []

    for rnd in range(Config.n_rounds):
        print(f"=== Round {rnd+1}/{Config.n_rounds} ===")

        local_models = []
        for cid in range(n_clients):
            lm = get_model()
            lm.load_state_dict(global_model.state_dict())
            lm = local_train(lm, client_train_datasets[cid], epochs=Config.local_epochs)
            local_models.append(lm)

        global_model = fedavg(local_models)

        # Evaluation:
        # - W_t: tail-only hidden test (welfare)
        # - M_t: head-only public validation (metric)
        # - overall test accuracy as an additional reference
        W_t = evaluate_model(global_model, hidden_tail_test_loader)
        M_t = evaluate_model(global_model, public_val_loader)
        overall_acc = evaluate_model(global_model, full_test_loader)

        W_history.append(W_t)
        M_history.append(M_t)
        overall_test_history.append(overall_acc)

        print(f"  Hidden tail test acc (W_t) : {W_t:.4f}")
        print(f"  Public head val acc (M_t)  : {M_t:.4f}")
        print(f"  Full test acc (ref)       : {overall_acc:.4f}")

    return W_history, M_history, overall_test_history


# ============================================================
# 9. Run aligned vs gaming experiments and summarize
# ============================================================

def tail_mean(vals, tail=10):
    vals = np.array(vals)
    if len(vals) < tail:
        return vals.mean()
    return vals[-tail:].mean()


if __name__ == "__main__":
    # aligned (no gaming)
    W_aligned, M_aligned, full_aligned = run_federated_experiment(gaming_frac=0.0)

    # gaming (30% gaming)
    W_gaming, M_gaming, full_gaming = run_federated_experiment(gaming_frac=Config.gaming_frac)

    W_align_mean = tail_mean(W_aligned)
    W_game_mean = tail_mean(W_gaming)
    M_align_mean = tail_mean(M_aligned)
    M_game_mean = tail_mean(M_gaming)
    full_align_mean = tail_mean(full_aligned)
    full_game_mean = tail_mean(full_gaming)

    if W_align_mean > 0:
        PoG = (W_align_mean - W_game_mean) / W_align_mean
    else:
        PoG = float("nan")

    print("\n=== Summary (last 10 rounds, head-metric vs tail-welfare) ===")
    print(f"W_align (tail) ≈ {W_align_mean:.3f}")
    print(f"W_game  (tail) ≈ {W_game_mean:.3f}")
    print(f"M_align (head) ≈ {M_align_mean:.3f}")
    print(f"M_game  (head) ≈ {M_game_mean:.3f}")
    print(f"Full test (aligned) ≈ {full_align_mean:.3f}")
    print(f"Full test (gaming)  ≈ {full_game_mean:.3f}")
    print(f"PoG (tail welfare)  ≈ {PoG:.3f}")