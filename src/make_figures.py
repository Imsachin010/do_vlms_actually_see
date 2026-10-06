"""
src/make_figures.py
Unified Figure Generation Pipeline.
Reads exclusively from results/ and analysis/ — never hand-edited or hardcoded values.

Usage:
    python src/make_figures.py --fig 1    # Overall Bias Rates
    python src/make_figures.py --fig 2    # Visual Salience vs. Bias Rate (Theil-Sen fit)
    python src/make_figures.py --fig 3    # Perception Bypass Ablation (Wilson CIs)
    python src/make_figures.py --fig 4    # Stacked Error Distribution
    python src/make_figures.py --fig 5    # Background Context Effect
    python src/make_figures.py --all      # Compile all figures (Figures 1-5)
"""

import argparse
from pathlib import Path
import subprocess
import sys

FIG_DIR = Path("assets")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_SCRIPTS = {
    1: ("Overall Bias Rates", "src/analysis/generate_figure_overall_bias_horizontal.py"),
    2: ("Visual Salience vs Bias Rate", "src/analysis/generate_figure_salience_scout.py"),
    3: ("Perception Bypass Ablation", "src/analysis/generate_figure_perception_ablation.py"),
    4: ("Error Distribution", "src/analysis/generate_figure_error_distribution.py"),
    5: ("Background Context Effect", "src/analysis/generate_figure_background_effect.py"),
}


def run_figure(fig_num: int):
    if fig_num not in FIGURE_SCRIPTS:
        print(f"[ERROR] Unknown figure number: {fig_num}. Choose from 1 to 5.")
        return

    name, script_path = FIGURE_SCRIPTS[fig_num]
    p = Path(script_path)
    if not p.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return

    print(f"\n>>> Compiling Figure {fig_num} ({name})...")
    res = subprocess.run([sys.executable, str(p)], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"    [OK] Figure {fig_num} compiled successfully into {FIG_DIR}/.")
    else:
        print(f"    [ERROR] Figure {fig_num} failed:\n{res.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Unified Publication Figure Generator")
    parser.add_argument("--fig", type=int, choices=[1, 2, 3, 4, 5], help="Figure number to generate (1-5)")
    parser.add_argument("--all", action="store_true", help="Generate all figures (1-5)")

    args = parser.parse_args()

    if args.fig:
        run_figure(args.fig)
    else:
        # Default or --all: run all figures
        for f_num in sorted(FIGURE_SCRIPTS.keys()):
            run_figure(f_num)

    print(f"\n[OK] Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
