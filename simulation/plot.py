import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_json_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_result_type(data):
    """Infer whether this JSON is tournament-style or bracket-style."""
    summary_rows = data.get("summary_rows", [])
    if not summary_rows:
        return "unknown"

    sample = summary_rows[0]
    if "win_rate" in sample and "avg_rank" in sample:
        return "tournament"
    if "champion_rate" in sample and "match_win_rate" in sample:
        return "bracket"
    return "unknown"


def load_latest_json(pattern):
    files = glob.glob(os.path.join(RESULT_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def resolve_inputs(args):
    """Resolve which files to process based on CLI options."""
    targets = []

    if args.json_file:
        targets.append(args.json_file)
        return targets

    if args.tournament_json:
        targets.append(args.tournament_json)
    if args.bracket_json:
        targets.append(args.bracket_json)

    if targets:
        return targets

    # Default behavior: generate BOTH using latest files when available.
    latest_tournament = load_latest_json("tournament_*.json")
    latest_bracket = load_latest_json("bracket_*.json")

    if latest_tournament:
        targets.append(latest_tournament)
    if latest_bracket:
        targets.append(latest_bracket)

    # Fallback for legacy/other naming.
    if not targets:
        latest_any = load_latest_json("*.json")
        if latest_any:
            targets.append(latest_any)

    return targets


def plot_tournament_win_rate(df, name):
    plt.figure(figsize=(10, 6))
    plt.bar(df["strategy"], df["win_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Rate")
    plt.title("Strategy Win Rate Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_winrate.png"))
    plt.close()


def plot_tournament_score_std(df, name):
    plt.figure(figsize=(10, 6))
    plt.bar(df["strategy"], df["avg_score"], yerr=df["std_score"], capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Score")
    plt.title("Average Score with Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_score_std.png"))
    plt.close()


def plot_rank_boxplot(data, name):
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


def plot_evolution_top_rate(data, name):
    evo_rows = pd.DataFrame(data.get("evolution_summary_rows", []))
    if evo_rows.empty:
        return

    evo_rows = evo_rows.sort_values("top_final_pop_rate", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(evo_rows["strategy"], evo_rows["top_final_pop_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Top-Final-Population Rate")
    plt.title("Evolution Winner Rate Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_evolution_top_rate.png"))
    plt.close()


def plot_bracket_champion_rate(df, name):
    ordered = df.sort_values("champion_rate", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(ordered["strategy"], ordered["champion_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Champion Rate")
    plt.title("Bracket Champion Rate Across Seeds")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_champion_rate.png"))
    plt.close()


def plot_bracket_podium_stacked(df, name):
    ordered = df.sort_values("champion_rate", ascending=False)
    x = ordered["strategy"]

    plt.figure(figsize=(11, 6))
    plt.bar(x, ordered["champion_rate"], label="Champion")
    plt.bar(
        x,
        ordered["runner_up_rate"],
        bottom=ordered["champion_rate"],
        label="Runner-up",
    )
    plt.bar(
        x,
        ordered["third_place_rate"],
        bottom=ordered["champion_rate"] + ordered["runner_up_rate"],
        label="Third place",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rate Across Seeds")
    plt.title("Podium Composition Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}_podium_stacked.png"))
    plt.close()


def plot_bracket_match_performance(df, name):
    ordered = df.sort_values("match_win_rate", ascending=False)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(ordered["strategy"], ordered["match_win_rate"], color="steelblue")
    ax1.set_ylabel("Match Win Rate", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(range(len(ordered["strategy"])))
    ax1.set_xticklabels(ordered["strategy"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(ordered["strategy"])),
        ordered["avg_match_score"],
        color="darkorange",
        marker="o",
    )
    ax2.set_ylabel("Average Match Score", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title("Bracket Match Performance: Win Rate and Average Score")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{name}_match_performance.png"))
    plt.close(fig)


def plot_file(json_file):
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    data = load_json_data(json_file)
    result_type = detect_result_type(data)

    if result_type == "tournament":
        df = pd.DataFrame(data["summary_rows"]).sort_values("win_rate", ascending=False)
        plot_tournament_win_rate(df, base_name)
        plot_tournament_score_std(df, base_name)
        if "rank_samples" in data:
            plot_rank_boxplot(data, base_name)
        plot_evolution_top_rate(data, base_name)
        return f"Tournament plots generated for {base_name}"

    if result_type == "bracket":
        df = pd.DataFrame(data["summary_rows"]).sort_values("champion_rate", ascending=False)
        plot_bracket_champion_rate(df, base_name)
        plot_bracket_podium_stacked(df, base_name)
        plot_bracket_match_performance(df, base_name)
        return f"Bracket plots generated for {base_name}"

    return f"Skipped unsupported JSON schema: {base_name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots from tournament and/or bracket JSON results. "
            "Default: latest tournament + latest bracket."
        )
    )
    parser.add_argument("--json-file", type=str, default=None, help="Single JSON file to process")
    parser.add_argument("--tournament-json", type=str, default=None, help="Tournament JSON file")
    parser.add_argument("--bracket-json", type=str, default=None, help="Bracket JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = resolve_inputs(args)

    if not targets:
        raise FileNotFoundError("No suitable JSON results found in results/")

    messages = []
    for json_file in targets:
        if not os.path.exists(json_file):
            messages.append(f"Skipped missing file: {json_file}")
            continue
        messages.append(plot_file(json_file))

    for msg in messages:
        print(msg)
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
