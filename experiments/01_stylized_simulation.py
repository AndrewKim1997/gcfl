import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


# ===================================================================
# 0. Utilities
# ===================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ===================================================================
# 1. Simulation configuration
# ===================================================================

@dataclass
class SimulationConfig:
    # basic structure
    n_clients: int = 100               # total number of clients
    T: int = 300                       # number of rounds
    gaming_frac: float = 0.3           # fraction of clients using gaming strategy

    # reward-related
    base_reward: float = 1.0           # scale from metric to reward
    reward_bias: float = 0.2           # baseline reward independent of metric

    # audit / penalty
    audit_budget: int = 5              # maximum number of clients audited per round
    alpha_penalty: float = 0.7         # penalty strength α when caught
    p_detect: float = 0.7              # probability of detecting gaming when audited

    # public / private metric mix
    rho_pub: float = 0.6               # weight on public metric (PB vs PC mix)
    noise_pub: float = 0.01            # noise on public metric
    noise_priv: float = 0.01           # noise on private metric

    # welfare impact
    gamma_welfare: float = 0.7         # coefficient γ: how strongly gaming harms welfare

    # participation cost / decision parameters
    base_cost_mean: float = 0.4        # mean participation cost across clients
    base_cost_std: float = 0.1         # std dev of participation cost
    participation_beta: float = 4.0    # logistic slope (larger → sharper threshold)
    participation_bias: float = 0.0    # logistic offset (baseline profit threshold)

    seed: Optional[int] = None         # random seed (for reproducibility)


# ===================================================================
# 2. Client initialization
# ===================================================================

def init_clients(cfg: SimulationConfig) -> Dict[str, Any]:
    """
    For each client, initialize:
    - type: 'honest' or 'gaming'
    - cost: participation cost
    """
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_clients
    n_gaming = int(np.round(cfg.gaming_frac * n))

    client_type = np.array(['honest'] * n, dtype=object)
    if n_gaming > 0:
        gaming_idx = rng.choice(n, size=n_gaming, replace=False)
        client_type[gaming_idx] = 'gaming'

    # participation cost: controlled by mean/std in config
    cost = rng.normal(loc=cfg.base_cost_mean,
                      scale=cfg.base_cost_std,
                      size=n)
    cost = np.clip(cost, 0.05, None)  # avoid negative or too small costs

    return {
        "client_type": client_type,
        "cost": cost,
        "rng": rng,
    }


# ===================================================================
# 3. Single-round simulation
# ===================================================================

def simulate_round(
    cfg: SimulationConfig,
    state: Dict[str, Any],
    M_prev: float
) -> Dict[str, Any]:

    client_type = state["client_type"]
    cost = state["cost"]
    rng = state["rng"]

    n = cfg.n_clients

    # ---------------------------------------------------------------
    # 3.1 Participation decision: logistic-based probabilistic join
    # ---------------------------------------------------------------
    approx_p_audit = cfg.audit_budget / max(n, 1)

    expected_penalty_gaming = cfg.alpha_penalty * approx_p_audit * cfg.p_detect
    expected_penalty_honest = 0.0

    profit = np.zeros(n)

    mask_g = (client_type == 'gaming')
    mask_h = (client_type == 'honest')

    profit[mask_g] = (
        cfg.reward_bias
        + cfg.base_reward * M_prev
        - expected_penalty_gaming
        - cost[mask_g]
    )
    profit[mask_h] = (
        cfg.reward_bias
        + cfg.base_reward * M_prev
        - expected_penalty_honest
        - cost[mask_h]
    )

    logits = cfg.participation_beta * (profit - cfg.participation_bias)
    p_participate = sigmoid(logits)

    participate = rng.random(size=n) < p_participate
    participants_idx = np.where(participate)[0]
    n_participants = len(participants_idx)

    # ---------------------------------------------------------------
    # 3.2 Number of honest / gaming participants and participation rate
    # ---------------------------------------------------------------
    if n_participants > 0:
        types_participants = client_type[participants_idx]
        H = int(np.sum(types_participants == 'honest'))
        G = int(np.sum(types_participants == 'gaming'))
    else:
        H = 0
        G = 0

    x_t = n_participants / n

    # ---------------------------------------------------------------
    # 3.3 Welfare and metric computation
    # ---------------------------------------------------------------
    if n == 0:
        W_t = 0.0
    else:
        W_t = (H - cfg.gamma_welfare * G) / n
        W_t = float(np.clip(W_t, 0.0, 1.0))

    delta_metric = 0.3
    noise_pub = rng.normal(0.0, cfg.noise_pub)
    noise_priv = rng.normal(0.0, cfg.noise_priv)

    M_pub = W_t + delta_metric * (G / max(n, 1)) + noise_pub
    M_priv = W_t + noise_priv

    M_t = cfg.rho_pub * M_pub + (1.0 - cfg.rho_pub) * M_priv
    M_t = float(np.clip(M_t, 0.0, 1.0))

    # ---------------------------------------------------------------
    # 3.4 Auditing and penalties
    # ---------------------------------------------------------------
    audited_idx = np.array([], dtype=int)
    penalized_idx = np.array([], dtype=int)

    if n_participants > 0 and cfg.audit_budget > 0:
        k = min(cfg.audit_budget, n_participants)
        audited_idx = rng.choice(participants_idx, size=k, replace=False)

        is_gaming_audited = (client_type[audited_idx] == 'gaming')
        detect_mask = rng.random(size=k) < (cfg.p_detect * is_gaming_audited.astype(float))
        penalized_idx = audited_idx[detect_mask]

    rewards = np.zeros(n)
    if n_participants > 0:
        rewards[participants_idx] = cfg.base_reward * M_t - cost[participants_idx]
        rewards[penalized_idx] -= cfg.alpha_penalty

    return {
        "W_t": W_t,
        "M_t": M_t,
        "x_t": x_t,
        "H_t": H,
        "G_t": G,
        "n_participants": n_participants,
        "audited_idx": audited_idx,
        "penalized_idx": penalized_idx,
        "rewards": rewards,
    }


# ===================================================================
# 4. Full simulation runner
# ===================================================================

def run_simulation(cfg: SimulationConfig) -> pd.DataFrame:
    """
    Run T rounds of simulation under a single policy configuration (cfg)
    and return a DataFrame of per-round logs.
    Logged fields: W_t, M_t, x_t, H_t, G_t, n_participants, n_audited, n_penalized.
    """
    state = init_clients(cfg)
    M_prev = 0.6  # initial metric

    logs: List[Dict[str, Any]] = []

    for t in range(cfg.T):
        round_result = simulate_round(cfg, state, M_prev)

        logs.append({
            "t": t,
            "W_t": round_result["W_t"],
            "M_t": round_result["M_t"],
            "x_t": round_result["x_t"],
            "H_t": round_result["H_t"],
            "G_t": round_result["G_t"],
            "n_participants": round_result["n_participants"],
            "n_audited": len(round_result["audited_idx"]),
            "n_penalized": len(round_result["penalized_idx"]),
        })

        M_prev = round_result["M_t"]

    df = pd.DataFrame(logs)
    return df


# ===================================================================
# 5. Helper for PoG estimation (aligned vs gaming)
# ===================================================================

def estimate_price_of_gaming(
    cfg_aligned: SimulationConfig,
    cfg_gaming: SimulationConfig,
    burn_in: int = 100
) -> Dict[str, Any]:
    """
    Compare two scenarios to approximate PoG:
    - cfg_aligned: gaming_frac = 0
    - cfg_gaming: gaming_frac > 0

    We approximate steady state by averaging W_t, x_t, M_t after 'burn_in' rounds.
    """

    df_align = run_simulation(cfg_aligned)
    df_game = run_simulation(cfg_gaming)

    mask_align = (df_align["t"] >= burn_in)
    mask_game = (df_game["t"] >= burn_in)

    W_align = df_align.loc[mask_align, "W_t"].mean()
    W_game = df_game.loc[mask_game, "W_t"].mean()

    x_align = df_align.loc[mask_align, "x_t"].mean()
    x_game = df_game.loc[mask_game, "x_t"].mean()

    M_align = df_align.loc[mask_align, "M_t"].mean()
    M_game = df_game.loc[mask_game, "M_t"].mean()

    if W_align <= 0:
        PoG = np.nan
    else:
        PoG = (W_align - W_game) / W_align

    return {
        "W_align": W_align,
        "W_game": W_game,
        "x_align": x_align,
        "x_game": x_game,
        "M_align": M_align,
        "M_game": M_game,
        "PoG": PoG,
        "cfg_aligned": asdict(cfg_aligned),
        "cfg_gaming": asdict(cfg_gaming),
        "df_aligned": df_align,
        "df_gaming": df_game,
    }


# ===================================================================
# 6. Experiment 1: sweep alpha_penalty vs PoG
# ===================================================================

def sweep_alpha(
    base_cfg: SimulationConfig,
    alphas: List[float],
    burn_in: int = 100,
) -> pd.DataFrame:
    """
    Vary alpha_penalty and measure PoG and participation.
    """
    rows = []

    for alpha in alphas:
        cfg_gaming = SimulationConfig(**{**asdict(base_cfg),
                                         "gaming_frac": base_cfg.gaming_frac,
                                         "alpha_penalty": alpha})
        cfg_aligned = SimulationConfig(**{**asdict(base_cfg),
                                          "gaming_frac": 0.0,
                                          "alpha_penalty": alpha})

        result = estimate_price_of_gaming(cfg_aligned, cfg_gaming, burn_in=burn_in)

        rows.append({
            "alpha_penalty": alpha,
            "W_align": result["W_align"],
            "W_game": result["W_game"],
            "x_align": result["x_align"],
            "x_game": result["x_game"],
            "M_align": result["M_align"],
            "M_game": result["M_game"],
            "PoG": result["PoG"],
        })

    return pd.DataFrame(rows)


# ===================================================================
# 7. Experiment 2: sweep rho_pub (public metric weight) vs PoG
# ===================================================================

def sweep_rho_pub(
    base_cfg: SimulationConfig,
    rhos: List[float],
    burn_in: int = 100,
) -> pd.DataFrame:
    """
    Vary rho_pub (weight on public metric) and measure PoG and M-W gap.
    """
    rows = []

    for rho in rhos:
        cfg_gaming = SimulationConfig(**{**asdict(base_cfg),
                                         "gaming_frac": base_cfg.gaming_frac,
                                         "rho_pub": rho})
        cfg_aligned = SimulationConfig(**{**asdict(base_cfg),
                                          "gaming_frac": 0.0,
                                          "rho_pub": rho})

        result = estimate_price_of_gaming(cfg_aligned, cfg_gaming, burn_in=burn_in)

        # M - W gap (metric inflation) under each scenario
        M_W_gap_align = result["M_align"] - result["W_align"]
        M_W_gap_game = result["M_game"] - result["W_game"]

        rows.append({
            "rho_pub": rho,
            "W_align": result["W_align"],
            "W_game": result["W_game"],
            "x_align": result["x_align"],
            "x_game": result["x_game"],
            "M_align": result["M_align"],
            "M_game": result["M_game"],
            "M_W_gap_align": M_W_gap_align,
            "M_W_gap_game": M_W_gap_game,
            "PoG": result["PoG"],
        })

    return pd.DataFrame(rows)


# ===================================================================
# 8. Experiment 3: participation vs alpha (resilience / stability)
# ===================================================================

def sweep_alpha_participation(
    base_cfg: SimulationConfig,
    alphas: List[float],
    burn_in: int = 100,
) -> pd.DataFrame:
    """
    Vary alpha_penalty and, under the gaming scenario, measure average participation (x_t).
    (We only track the gaming case here rather than aligned vs gaming.)
    """
    rows = []

    for alpha in alphas:
        cfg_gaming = SimulationConfig(**{**asdict(base_cfg),
                                         "alpha_penalty": alpha})

        df_game = run_simulation(cfg_gaming)
        mask_game = (df_game["t"] >= burn_in)

        x_mean = df_game.loc[mask_game, "x_t"].mean()
        W_mean = df_game.loc[mask_game, "W_t"].mean()

        rows.append({
            "alpha_penalty": alpha,
            "x_game_mean": x_mean,
            "W_game_mean": W_mean,
        })

    return pd.DataFrame(rows)


# ===================================================================
# 9. Simple usage example (only when run as a script)
# ===================================================================

if __name__ == "__main__":
    # Base configuration (near the "bad policy" region used in the discussion)
    base_cfg = SimulationConfig(
        n_clients=100,
        T=300,
        gaming_frac=0.3,
        base_reward=1.0,
        reward_bias=0.2,
        audit_budget=10,
        alpha_penalty=0.7,
        p_detect=0.7,
        rho_pub=0.6,
        noise_pub=0.01,
        noise_priv=0.01,
        gamma_welfare=0.7,
        base_cost_mean=0.4,
        base_cost_std=0.1,
        participation_beta=4.0,
        participation_bias=0.0,
        seed=42,
    )

    print("=== Single simulation example (gaming scenario) ===")
    df_example = run_simulation(base_cfg)
    print(df_example.head())
    print(df_example.tail())

    print("\n=== PoG (aligned vs gaming) example ===")
    cfg_aligned = SimulationConfig(**{**asdict(base_cfg), "gaming_frac": 0.0})
    result = estimate_price_of_gaming(cfg_aligned, base_cfg, burn_in=100)
    print(f"W_align ≈ {result['W_align']:.3f}")
    print(f"W_game  ≈ {result['W_game']:.3f}")
    print(f"x_align ≈ {result['x_align']:.3f}")
    print(f"x_game  ≈ {result['x_game']:.3f}")
    print(f"PoG     ≈ {result['PoG']:.3f}")

    print("\n=== Experiment 1: PoG vs alpha_penalty ===")
    alphas = [0.3, 0.5, 0.7, 1.0, 1.5]
    df_alpha = sweep_alpha(base_cfg, alphas, burn_in=100)
    print(df_alpha)

    print("\n=== Experiment 2: PoG vs rho_pub ===")
    rhos = [1.0, 0.8, 0.6, 0.4, 0.2]
    df_rho = sweep_rho_pub(base_cfg, rhos, burn_in=100)
    print(df_rho)

    print("\n=== Experiment 3: participation vs alpha (gaming only) ===")
    df_alpha_part = sweep_alpha_participation(base_cfg, alphas, burn_in=100)
    print(df_alpha_part)