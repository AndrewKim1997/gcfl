"""Small, illustrative PoG simulation; this is not a paper experiment."""

from dataclasses import replace

from gcfl import SimulationConfig, estimate_price_of_gaming


base = SimulationConfig(n_clients=20, T=30, seed=42)
aligned = replace(base, gaming_frac=0.0)
gaming = replace(base, gaming_frac=0.3)
result = estimate_price_of_gaming(aligned, gaming, burn_in=10)

print(f"Aligned welfare: {result['W_align']:.3f}")
print(f"Gaming welfare:  {result['W_game']:.3f}")
print(f"Price of gaming: {result['PoG']:.3f}")
