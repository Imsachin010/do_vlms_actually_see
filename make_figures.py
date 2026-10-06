"""
make_figures.py
Unified pipeline runner to regenerate all publication figures from raw CSV logs.

Figures produced:
  - Figure 1: Overall Bias Rates Across Architectures
  - Figure 2: Bias Rate vs. Visual Feature Salience (Llama-4-Scout)
  - Figure 3: Comprehensive Perception Bypass Ablation (14 Configurations)
  - Figure 4: Stacked Error Distribution Across Architectures
  - Figure 5: Background Context Effect on Bias Rate

Usage:
    python make_figures.py
"""

import sys
import subprocess
from pathlib import Path

FIG_DIR = Path("assets")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("Figure 1", "src/analysis/generate_figure_overall_bias_horizontal.py"),
    ("Figure 2", "src/analysis/generate_figure_salience_scout.py"),
    ("Figure 3", "src/analysis/generate_figure_perception_ablation.py"),
    ("Figure 4", "src/analysis/generate_figure_error_distribution.py"),
    ("Figure 5", "src/analysis/generate_figure_background_effect.py"),
]

def make_all_figures():
    print("=" * 70)
    print("REGENERATING ALL PUBLICATION FIGURES FROM RAW AUDITED LOGS")
    print("=" * 70)

    for name, script in SCRIPTS:
        p = Path(script)
        if not p.exists():
            print(f"[SKIP] Script {script} not found.")
            continue

        print(f"\n>>> Running {name} generator: {script} ...")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"    [OK] {name} generated successfully into {FIG_DIR}/.")
        else:
            print(f"    [ERROR] {name} failed: {res.stderr}")

    print("\n" + "=" * 70)
    print(f"All figures compiled and saved to {FIG_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    make_all_figures()
