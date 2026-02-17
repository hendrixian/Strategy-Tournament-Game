import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_latest_json():
    files = glob.glob(os.path.join(RESULT_DIR, "*.json"))
    if not files:
        raise FileNotFoundError("No JSON results found in results/")
    return max(files, key=os.path.getmtime)


def load_dataframe(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["summary_rows"])
    df = df.sort_values("win_rate", ascending=False)
    return df


# -------------------------------------------------
# 1) WIN RATE BAR CHART
# -------------------------------------------------
def plot_win_rate(df, name):
    plt.figure(figsize=(10, 6))
    plt.bar(df["strategy"], df["win_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Rate")
    plt.title("Strategy Win Rate Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_winrate.png"))
    plt.close()


# -------------------------------------------------
# 2) SCORE ± STD ERROR BAR
# -------------------------------------------------
def plot_score_std(df, name):
    plt.figure(figsize=(10, 6))
    plt.bar(df["strategy"], df["avg_score"], yerr=df["std_score"], capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Score")
    plt.title("Average Score with Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_score_std.png"))
    plt.close()


# -------------------------------------------------
# 3) RANK BOXPLOT (IMPORTANT)
# -------------------------------------------------
def plot_rank_boxplot(json_file, name):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rank_samples = data["rank_samples"]

    strategies = list(rank_samples.keys())
    values = [rank_samples[s] for s in strategies]

    plt.figure(figsize=(10, 6))
    plt.boxplot(values, labels=strategies)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rank (lower is better)")
    plt.title("Rank Distribution Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_rank_boxplot.png"))
    plt.close()


def main():
    json_file = load_latest_json()
    base_name = os.path.splitext(os.path.basename(json_file))[0]

    df = load_dataframe(json_file)

    plot_win_rate(df, base_name)
    plot_score_std(df, base_name)
    plot_rank_boxplot(json_file, base_name)

    print("Plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
