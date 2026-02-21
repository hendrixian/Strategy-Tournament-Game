import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_json_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_result_type(data):
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
    targets = []

    if args.json_file:
        return [args.json_file]

    if args.tournament_json:
        targets.append(args.tournament_json)
    if args.bracket_json:
        targets.append(args.bracket_json)

    if targets:
        return targets

    latest_tournament = load_latest_json("tournament_*.json")
    latest_bracket = load_latest_json("bracket_*.json")

    if latest_tournament:
        targets.append(latest_tournament)
    if latest_bracket:
        targets.append(latest_bracket)

    if not targets:
        latest_any = load_latest_json("*.json")
        if latest_any:
            targets.append(latest_any)

    return targets


def parse_config_from_name(base_name, result_type):
    if result_type == "tournament":
        m = re.match(
            r"^tournament_(?P<seeds>\d+)s_(?P<rounds>\d+)r_noise(?P<noise>[^_]+)(?:_gen(?P<gen>[^_]+)_mut(?P<mut>[^_]+))?_",
            base_name,
        )
        if m:
            parts = [
                "Type: Round-robin",
                f"Seeds: {m.group('seeds')}",
                f"Rounds/Match: {m.group('rounds')}",
                f"Noise: {m.group('noise')}",
            ]
            if m.group("gen") is not None and m.group("mut") is not None:
                parts.append(f"Generations: {m.group('gen')}")
                parts.append(f"Mutation: {m.group('mut')}")
            return " | ".join(parts)

    if result_type == "bracket":
        m = re.match(
            r"^bracket_(?P<btype>single|double)_(?P<seeds>\d+)s_(?P<rounds>\d+)r_noise(?P<noise>[^_]+)_",
            base_name,
        )
        if m:
            return " | ".join(
                [
                    f"Type: {m.group('btype').title()} Elimination",
                    f"Seeds: {m.group('seeds')}",
                    f"Rounds/Match: {m.group('rounds')}",
                    f"Noise: {m.group('noise')}",
                ]
            )

    return "Configuration: Unknown"


def save_tournament_combo(df, data, base_name, game_name):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1) Win rate
    axes[0].bar(df["strategy"], df["win_rate"])
    axes[0].set_title("Strategy Win Rate Across Seeds")
    axes[0].set_ylabel("Win Rate")
    axes[0].tick_params(axis="x", rotation=45)

    # 2) Avg score +/- std
    axes[1].bar(df["strategy"], df["avg_score"], yerr=df["std_score"], capsize=4)
    axes[1].set_title("Average Score with Standard Deviation")
    axes[1].set_ylabel("Average Score")
    axes[1].tick_params(axis="x", rotation=45)

    # 3) Rank boxplot
    rank_samples = data.get("rank_samples", {})
    strategies = list(rank_samples.keys())
    values = [rank_samples[s] for s in strategies]
    if strategies:
        axes[2].boxplot(values, labels=strategies)
    axes[2].set_title("Rank Distribution Across Seeds")
    axes[2].set_ylabel("Rank (lower is better)")
    axes[2].tick_params(axis="x", rotation=45)

    config_text = parse_config_from_name(base_name, "tournament")
    fig.suptitle(f"{game_name} | Tournament Summary\n{config_text}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    out = os.path.join(PLOT_DIR, f"{base_name}_combined.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_bracket_combo(df, base_name, game_name):
    ordered_champ = df.sort_values("champion_rate", ascending=False)
    ordered_match = df.sort_values("match_win_rate", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1) Champion rate
    axes[0].bar(ordered_champ["strategy"], ordered_champ["champion_rate"])
    axes[0].set_title("Bracket Champion Rate Across Seeds")
    axes[0].set_ylabel("Champion Rate")
    axes[0].tick_params(axis="x", rotation=45)

    # 2) Podium stacked
    x = ordered_champ["strategy"]
    axes[1].bar(x, ordered_champ["champion_rate"], label="Champion")
    axes[1].bar(
        x,
        ordered_champ["runner_up_rate"],
        bottom=ordered_champ["champion_rate"],
        label="Runner-up",
    )
    axes[1].bar(
        x,
        ordered_champ["third_place_rate"],
        bottom=ordered_champ["champion_rate"] + ordered_champ["runner_up_rate"],
        label="Third place",
    )
    axes[1].set_title("Podium Composition Across Seeds")
    axes[1].set_ylabel("Rate Across Seeds")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(loc="upper right")

    # 3) Match performance (bar + line)
    x_idx = range(len(ordered_match["strategy"]))
    axes[2].bar(x_idx, ordered_match["match_win_rate"], color="steelblue")
    axes[2].set_xticks(list(x_idx))
    axes[2].set_xticklabels(ordered_match["strategy"], rotation=45, ha="right")
    axes[2].set_ylabel("Match Win Rate", color="steelblue")
    axes[2].tick_params(axis="y", labelcolor="steelblue")
    axes[2].set_title("Bracket Match Performance")

    ax2 = axes[2].twinx()
    ax2.plot(x_idx, ordered_match["avg_match_score"], color="darkorange", marker="o")
    ax2.set_ylabel("Average Match Score", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    config_text = parse_config_from_name(base_name, "bracket")
    fig.suptitle(f"{game_name} | Bracket Summary\n{config_text}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    out = os.path.join(PLOT_DIR, f"{base_name}_combined.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_file(json_file, game_name):
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    data = load_json_data(json_file)
    result_type = detect_result_type(data)

    if result_type == "tournament":
        df = pd.DataFrame(data["summary_rows"]).sort_values("win_rate", ascending=False)
        actual_game_name = data.get("game_name", game_name)
        out = save_tournament_combo(df, data, base_name, actual_game_name)
        return f"Tournament combined plot generated: {out}"

    if result_type == "bracket":
        df = pd.DataFrame(data["summary_rows"]).sort_values("champion_rate", ascending=False)
        out = save_bracket_combo(df, base_name, game_name)
        return f"Bracket combined plot generated: {out}"

    return f"Skipped unsupported JSON schema: {base_name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one combined 3-panel image per tournament/bracket JSON. "
            "Default: latest tournament + latest bracket."
        )
    )
    parser.add_argument("--json-file", type=str, default=None, help="Single JSON file to process")
    parser.add_argument("--tournament-json", type=str, default=None, help="Tournament JSON file")
    parser.add_argument("--bracket-json", type=str, default=None, help="Bracket JSON file")
    parser.add_argument(
        "--game-name",
        type=str,
        default="Prisoner's Dilemma",
        help="Game name shown in combined chart title",
    )
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
        messages.append(plot_file(json_file, args.game_name))

    for msg in messages:
        print(msg)
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
