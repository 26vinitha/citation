import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================
RESULTS_JSON = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\model_all_splits_results_multiyear_n10.json"
DATA_CSV = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\mlp_training_data_fulltext_1993_2003_n10.csv"
OUTPUT_DIR = r"C:\Users\vinipumba\Desktop\citation-count-simple\try 2\comparison_multiyear_n10"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"], data["metadata"]


def build_summary_table(results):
    rows = []
    for r in results:
        if r.get("failed"):
            continue
        rows.append({
            "model_name": r["model_name"],
            "architecture": r["model_type"],
            "feature_set": r["feature_set"],
            "config": r["config_name"],
            "split": r["split_name"],
            "test_MAE": r["metrics"]["test"]["mae"],
            "test_RMSE": r["metrics"]["test"]["rmse"],
            "test_R2": r["metrics"]["test"]["r2"],
        })
    return pd.DataFrame(rows)


def compute_naive_baseline():
    df = pd.read_csv(DATA_CSV, dtype={"article_id": str})
    y = df["citations_2022_target"].values.astype(float)
    naive_pred = df["citations_2021"].values.astype(float)
    mae = np.mean(np.abs(y - naive_pred))
    rmse = np.sqrt(np.mean((y - naive_pred) ** 2))
    ss_res = np.sum((y - naive_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("Loading results...")
    results, metadata = load_results()
    print(f"Total models: {len(results)}")

    table = build_summary_table(results)
    table_sorted = table.sort_values("test_RMSE")
    table_sorted.to_csv(f"{OUTPUT_DIR}/all_models_comparison.csv", index=False)

    naive = compute_naive_baseline()
    print(f"\nNaive persistence baseline: MAE={naive['MAE']:.4f}  RMSE={naive['RMSE']:.4f}  R2={naive['R2']:.4f}")

    best = table_sorted.iloc[0]
    print("\n" + "=" * 70)
    print("BEST MODEL (lowest test RMSE)")
    print("=" * 70)
    print(best.to_string())
    beats_naive = best["test_RMSE"] < naive["RMSE"]
    print(f"\nBeats naive persistence baseline (RMSE {naive['RMSE']:.4f})? {'YES' if beats_naive else 'NO'}")

    # ---- best model PER architecture -- this is what answers "what about the other models" ----
    best_per_arch = table.loc[table.groupby("architecture")["test_RMSE"].idxmin()].sort_values("test_RMSE")
    best_per_arch.to_csv(f"{OUTPUT_DIR}/best_model_per_architecture.csv", index=False)
    print("\nBest model per architecture (all 5, including RNN):")
    print(best_per_arch[["architecture", "model_name", "test_MAE", "test_RMSE", "test_R2"]].to_string(index=False))

    # ================================================================
    # GRAPH 1
    # ================================================================
    plt.figure(figsize=(8, 5))
    archs = best_per_arch["architecture"].tolist()
    r2_vals = best_per_arch["test_R2"].tolist()
    colors = ["#2ca02c" if v > naive["R2"] else "#d62728" for v in r2_vals]
    plt.bar(archs, r2_vals, color=colors)
    plt.axhline(naive["R2"], color="black", linestyle="--", linewidth=1.5,
                label=f"Naive persistence baseline (R\u00b2={naive['R2']:.3f})")
    plt.ylabel("Best Test R\u00b2")
    plt.title("Best R\u00b2 by Architecture vs. Naive Baseline (Full Corpus)\n(green = beats baseline, red = does not)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/r2_by_architecture_vs_baseline.png", dpi=150)
    plt.close()

    # ================================================================
    # GRAPH 2
    # ================================================================
    plt.figure(figsize=(8, 5))
    rmse_vals = best_per_arch["test_RMSE"].tolist()
    colors = ["#2ca02c" if v < naive["RMSE"] else "#d62728" for v in rmse_vals]
    plt.bar(archs, rmse_vals, color=colors)
    plt.axhline(naive["RMSE"], color="black", linestyle="--", linewidth=1.5,
                label=f"Naive persistence baseline (RMSE={naive['RMSE']:.3f})")
    plt.ylabel("Best Test RMSE (lower is better)")
    plt.title("Best RMSE by Architecture vs. Naive Baseline (Full Corpus)\n(green = beats baseline, red = does not)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rmse_by_architecture_vs_baseline.png", dpi=150)
    plt.close()

    # ================================================================
    # GRAPH 3
    # ================================================================
    plt.figure(figsize=(9, 5))
    arch_order = table.groupby("architecture")["test_R2"].median().sort_values(ascending=False).index.tolist()
    data_by_arch = [table[table["architecture"] == a]["test_R2"].values for a in arch_order]
    try:
        plt.boxplot(data_by_arch, tick_labels=arch_order)
    except TypeError:
        plt.boxplot(data_by_arch, labels=arch_order)
    plt.axhline(naive["R2"], color="black", linestyle="--", linewidth=1.5,
                label=f"Naive baseline (R\u00b2={naive['R2']:.3f})")
    plt.ylabel("Test R\u00b2 (all configs/splits)")
    plt.title("R\u00b2 Distribution by Architecture (Full Corpus)\n(across all feature sets, configs, and train/test splits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/r2_distribution_boxplot.png", dpi=150)
    plt.close()

    # ================================================================
    # GRAPH 4
    # ================================================================
    plt.figure(figsize=(8, 5))
    feat_order = ["2021_only", "2020_2021", "full_history"]
    feat_r2 = [table[table["feature_set"] == f]["test_R2"].mean() for f in feat_order]
    plt.bar(feat_order, feat_r2, color="#1f77b4")
    plt.axhline(naive["R2"], color="black", linestyle="--", linewidth=1.5,
                label=f"Naive baseline (R\u00b2={naive['R2']:.3f})")
    plt.ylabel("Mean Test R\u00b2 (across all architectures/configs/splits)")
    plt.title("Effect of Feature Set on Prediction Accuracy (Full Corpus)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_set_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("TOP 10 MODELS OVERALL (by test RMSE)")
    print("=" * 70)
    print(table_sorted.head(10)[["model_name", "test_RMSE", "test_R2"]].to_string(index=False))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("  - all_models_comparison.csv       (full model table, sorted)")
    print("  - best_model_per_architecture.csv (5 rows, one per architecture -- includes RNN)")
    print("  - r2_by_architecture_vs_baseline.png")
    print("  - rmse_by_architecture_vs_baseline.png")
    print("  - r2_distribution_boxplot.png")
    print("  - feature_set_comparison.png")


if __name__ == "__main__":
    main()
