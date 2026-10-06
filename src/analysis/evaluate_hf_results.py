import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HF_RESULTS_DIR = os.path.join("data", "raw", "results_hf")
OUT_DIR = os.path.join("results", "hf_evaluation")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")


def ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_hf_results() -> pd.DataFrame:
    files = sorted(glob(os.path.join(HF_RESULTS_DIR, "*_results.csv")))
    if not files:
        raise FileNotFoundError(f"No HF result CSVs found in: {HF_RESULTS_DIR}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Keep latest row per model+image in case of reruns.
    all_df["_row_id"] = range(len(all_df))
    all_df = (
        all_df.sort_values("_row_id")
        .drop_duplicates(subset=["model", "image"], keep="last")
        .drop(columns=["_row_id"])
    )

    # Numeric safety.
    all_df["predicted"] = pd.to_numeric(all_df["predicted"], errors="coerce")
    all_df["true_legs"] = pd.to_numeric(all_df["true_legs"], errors="coerce")
    all_df["is_biased"] = pd.to_numeric(all_df["is_biased"], errors="coerce")
    all_df["has_prediction"] = all_df["predicted"].notna().astype(int)
    all_df["abs_error"] = (all_df["predicted"] - all_df["true_legs"]).abs()

    return all_df


def make_tables(df: pd.DataFrame) -> None:
    model_summary = (
        df.groupby("model")
        .agg(
            n_images=("image", "count"),
            prediction_coverage=("has_prediction", "mean"),
            bias_rate=("is_biased", "mean"),
            mean_abs_error=("abs_error", "mean"),
            api_error_rate=("status", lambda s: (s == "api_error").mean()),
            parse_failed_rate=("status", lambda s: (s == "parse_failed").mean()),
            empty_output_rate=("status", lambda s: (s == "empty_output").mean()),
        )
        .reset_index()
        .sort_values("bias_rate", ascending=False)
    )
    model_summary.to_csv(os.path.join(TABLES_DIR, "hf_model_summary.csv"), index=False)

    bg_summary = (
        df.groupby(["model", "background"])
        .agg(
            n_images=("image", "count"),
            prediction_coverage=("has_prediction", "mean"),
            bias_rate=("is_biased", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["model", "background"])
    )
    bg_summary.to_csv(os.path.join(TABLES_DIR, "hf_background_summary.csv"), index=False)

    concept_summary = (
        df.groupby(["model", "concept"])
        .agg(
            n_images=("image", "count"),
            prediction_coverage=("has_prediction", "mean"),
            bias_rate=("is_biased", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["model", "bias_rate"], ascending=[True, False])
    )
    concept_summary.to_csv(os.path.join(TABLES_DIR, "hf_concept_summary.csv"), index=False)

    status_summary = (
        df.groupby(["model", "status"]).size().reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )
    status_summary.to_csv(os.path.join(TABLES_DIR, "hf_status_summary.csv"), index=False)


def make_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Bias rate by model.
    model_bias = df.groupby("model", as_index=False)["is_biased"].mean()
    model_bias = model_bias.sort_values("is_biased", ascending=False)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=model_bias, x="model", y="is_biased", hue="model", legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("Bias Rate")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hf_bias_rate_by_model.png"), dpi=300)
    plt.close()

    # 2) Prediction coverage by model.
    coverage = df.groupby("model", as_index=False)["has_prediction"].mean()
    coverage = coverage.sort_values("has_prediction", ascending=True)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=coverage, x="model", y="has_prediction", hue="model", legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("Prediction Coverage")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hf_prediction_coverage_by_model.png"), dpi=300)
    plt.close()

    # 3) Background effect.
    bg = df.groupby(["model", "background"], as_index=False)["is_biased"].mean()
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=bg, x="model", y="is_biased", hue="background")
    ax.set_xlabel("")
    ax.set_ylabel("Bias Rate")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hf_background_bias_effect.png"), dpi=300)
    plt.close()

    # 4) Status distribution.
    status = df.groupby(["model", "status"]).size().reset_index(name="count")
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=status, x="model", y="count", hue="status")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hf_status_distribution.png"), dpi=300)
    plt.close()


def write_quick_summary(df: pd.DataFrame) -> None:
    lines = []
    lines.append("HF Evaluation Summary")
    lines.append("====================")
    lines.append(f"Total rows (deduplicated latest per model-image): {len(df)}")
    lines.append(f"Models: {', '.join(sorted(df['model'].dropna().unique()))}")
    lines.append("")

    m = (
        df.groupby("model")
        .agg(
            n_images=("image", "count"),
            coverage=("has_prediction", "mean"),
            bias_rate=("is_biased", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    for _, row in m.iterrows():
        lines.append(
            f"- {row['model']}: n={int(row['n_images'])}, "
            f"coverage={row['coverage']:.3f}, bias={row['bias_rate']:.3f}, mae={row['mae']:.3f}"
        )

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run() -> None:
    ensure_dirs()
    df = load_hf_results()
    make_tables(df)
    make_figures(df)
    write_quick_summary(df)
    print(f"HF evaluation complete. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    run()
