"""Extract the original GCFL experiment cells without changing their code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive" / "GCFL_main.ipynb"
EXPECTED_SHA256 = "da91b43ceef19ea9bb66e254e94d651be419efa878025361b1a29670ab179d8b"

# Zero-based cell indexes in the original notebook. Cell 11 is a Colab !pip
# setup command; requirements.txt supplies those dependencies outside Colab.
EXPERIMENTS = {
    "01_stylized_simulation.py": 1,
    "02_real_world_fl.py": 3,
    "03_estimator_reliability.py": 5,
    "04_noise_and_auditability.py": 7,
    "05_high_alignment_metrics.py": 9,
    "06_modern_attack.py": 12,
}

# The first 284 source lines of cell 1 contain the simulator and its PoG
# estimator. Keeping the API's underlying calculations as a source extract
# makes their relationship to the notebook explicit.
STYLIZED_CORE_LINES = 284


def extract(*, check: bool = False) -> None:
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Original notebook hash changed: {digest}")

    cells = json.loads(raw)["cells"]
    destination = ROOT / "experiments"
    if not check:
        destination.mkdir(parents=True, exist_ok=True)

    for filename, index in EXPERIMENTS.items():
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"Cell {index} is not code")
        code = "".join(cell["source"])
        compile(code, filename, "exec")
        target = destination / filename
        encoded = code.encode("utf-8")
        if check:
            if not target.is_file() or target.read_bytes() != encoded:
                raise RuntimeError(f"Experiment differs from source cell {index}: {target}")
        else:
            target.write_bytes(encoded)

    core = "".join(cells[1]["source"][:STYLIZED_CORE_LINES])
    compile(core, "stylized.py", "exec")
    core_path = ROOT / "src" / "gcfl" / "stylized.py"
    if check:
        if not core_path.is_file() or core_path.read_bytes() != core.encode("utf-8"):
            raise RuntimeError(f"Reusable simulator differs from original cell: {core_path}")
    else:
        core_path.parent.mkdir(parents=True, exist_ok=True)
        core_path.write_bytes(core.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare existing scripts with original cells")
    extract(check=parser.parse_args().check)
