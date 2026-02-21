"""Run repeated bracket tournament simulations across many seeds."""

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import payoff

# Make project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bracket import BracketTournament
from payoff import PayoffMatrix
from strategies import get_all_strategies

GAME_TYPE = PayoffMatrix.PRISONERS_DILEMMA


class Tee:
    """Write output to both terminal and a file."""

    def __init__(self, filename: str):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@dataclass
class SeedRunResult:
    seed: int
    champion: str
    runner_up: str
    third_place: str
    total_rounds: int
    match_history: List[Dict[str, Any]]


def run_bracket_for_seed(seed: int, rounds: int, noise: float, bracket_type: str) -> SeedRunResult:
    """Run one bracket tournament under a single seed."""
    random.seed(seed)
    np.random.seed(seed)

    strategies = get_all_strategies()
    payoff = PayoffMatrix(GAME_TYPE)
    bracket = BracketTournament(
        strategies=strategies,
        payoff_matrix=payoff,
        rounds_per_match=rounds,
        noise=noise,
        bracket_type=bracket_type,
    )
    results = bracket.run_tournament()

    return SeedRunResult(
        seed=seed,
        champion=results.get("champion"),
        runner_up=results.get("runner_up"),
        third_place=results.get("third_place"),
        total_rounds=results.get("total_rounds", 0),
        match_history=results.get("match_history", []),
    )


def summarize_runs(runs: List[SeedRunResult], game_name: str) -> Dict[str, Any]:
    """Compute aggregate statistics across all runs."""
    strategy_names = sorted([s.name for s in get_all_strategies()])

    champion_counts = {name: 0 for name in strategy_names}
    runner_up_counts = {name: 0 for name in strategy_names}
    third_place_counts = {name: 0 for name in strategy_names}
    match_win_counts = {name: 0 for name in strategy_names}
    match_play_counts = {name: 0 for name in strategy_names}
    score_samples = {name: [] for name in strategy_names}
    rounds_samples: List[int] = []

    for run in runs:
        if run.champion in champion_counts:
            champion_counts[run.champion] += 1
        if run.runner_up in runner_up_counts:
            runner_up_counts[run.runner_up] += 1
        if run.third_place in third_place_counts:
            third_place_counts[run.third_place] += 1
        rounds_samples.append(run.total_rounds)

        for match in run.match_history:
            s1 = match.get("strategy1")
            s2 = match.get("strategy2")
            winner = match.get("winner")
            score1 = match.get("score1")
            score2 = match.get("score2")

            if s1 in match_play_counts:
                match_play_counts[s1] += 1
                if isinstance(score1, (int, float)):
                    score_samples[s1].append(float(score1))
            if s2 in match_play_counts:
                match_play_counts[s2] += 1
                if isinstance(score2, (int, float)):
                    score_samples[s2].append(float(score2))
            if winner in match_win_counts:
                match_win_counts[winner] += 1

    total_runs = len(runs)
    strategy_rows = []
    for name in strategy_names:
        plays = match_play_counts[name]
        wins = match_win_counts[name]
        strategy_rows.append(
            {
                "strategy": name,
                "champion_count": champion_counts[name],
                "champion_rate": champion_counts[name] / total_runs,
                "runner_up_count": runner_up_counts[name],
                "runner_up_rate": runner_up_counts[name] / total_runs,
                "third_place_count": third_place_counts[name],
                "third_place_rate": third_place_counts[name] / total_runs,
                "match_wins": wins,
                "match_plays": plays,
                "match_win_rate": (wins / plays) if plays > 0 else 0.0,
                "avg_match_score": statistics.mean(score_samples[name]) if score_samples[name] else 0.0,
                "std_match_score": statistics.pstdev(score_samples[name]) if score_samples[name] else 0.0,
            }
        )

    strategy_rows.sort(
        key=lambda row: (
            -row["champion_count"],
            -row["match_win_rate"],
            -row["avg_match_score"],
        )
    )

    return {
        "total_runs": total_runs,
        "summary_rows": strategy_rows,
        "game_name": game_name,
        "champions_by_seed": [(run.seed, run.champion) for run in runs],
        "runner_ups_by_seed": [(run.seed, run.runner_up) for run in runs],
        "third_places_by_seed": [(run.seed, run.third_place) for run in runs],
        "avg_total_rounds": statistics.mean(rounds_samples) if rounds_samples else 0.0,
        "std_total_rounds": statistics.pstdev(rounds_samples) if rounds_samples else 0.0,
    }


def print_run_details(run: SeedRunResult, detail_level: str) -> None:
    """Print detailed information for one run."""
    print("=" * 90)
    print(f"Run seed={run.seed}")
    print(f"Champion: {run.champion}")
    print(f"Runner-up: {run.runner_up}")
    print(f"Third place: {run.third_place}")
    print(f"Total bracket rounds: {run.total_rounds}")

    if detail_level in {"match", "round"}:
        print("Match-by-match outcomes:")
        for idx, match in enumerate(run.match_history, start=1):
            s1 = match.get("strategy1")
            s2 = match.get("strategy2")
            w = match.get("winner")
            print(
                f"  Match {idx:02d}: {s1} vs {s2} | "
                f"scores=({match.get('score1', 0):.2f}, {match.get('score2', 0):.2f}) | "
                f"winner={w}"
            )

            if detail_level == "round":
                for round_idx, step in enumerate(match.get("history", []), start=1):
                    if len(step) == 4:
                        m1, m2, p1, p2 = step
                        print(
                            f"    Round {round_idx:02d}: {s1}:{m1} {s2}:{m2} "
                            f"-> payoff=({p1:.2f}, {p2:.2f})"
                        )


def print_summary(summary: Dict[str, Any]) -> None:
    """Print aggregate results over all seeds."""
    print("\n" + "#" * 90)
    print("Aggregate multi-seed bracket summary")
    print("#" * 90)
    print(f"Total runs: {summary['total_runs']}")
    print(
        "Average total rounds: "
        f"{summary['avg_total_rounds']:.3f} +/- {summary['std_total_rounds']:.3f}"
    )

    print("\nChampion by seed:")
    for seed, champion in summary["champions_by_seed"]:
        print(f"  seed={seed:4d} -> champion={champion}")

    print("\nPer-strategy aggregate metrics:")
    print(
        "  (champion/runner-up/third rates, match win rate, average match score +/- std)"
    )
    for row in summary["summary_rows"]:
        print(
            f"  {row['strategy']:20s} "
            f"champ={row['champion_count']:3d} ({row['champion_rate']:.2%}) "
            f"runner={row['runner_up_count']:3d} ({row['runner_up_rate']:.2%}) "
            f"third={row['third_place_count']:3d} ({row['third_place_rate']:.2%}) "
            f"m_win_rate={row['match_win_rate']:.3f} "
            f"avg_score={row['avg_match_score']:.3f} +/- {row['std_match_score']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated bracket tournaments over many seeds and report outcomes."
    )
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds/runs")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per match")
    parser.add_argument("--noise", type=float, default=0.20, help="Noise level [0,1]")
    parser.add_argument(
        "--bracket-type",
        choices=["single", "double"],
        default="single",
        help="Bracket format",
    )
    parser.add_argument(
        "--detail-level",
        choices=["summary", "match", "round"],
        default="summary",
        help="Output detail granularity for each run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be > 0")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if not 0.0 <= args.noise <= 1.0:
        raise ValueError("--noise must be between 0 and 1")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        results_dir,
        f"bracket_{args.bracket_type}_{args.num_seeds}s_{args.rounds}r_noise{args.noise}_{timestamp}.txt",
    )
    sys.stdout = Tee(outfile)
    print(f"Output saved to: {outfile}")

    print("Running multi-seed bracket experiment")
    print(
        "Configuration: "
        f"num_seeds={args.num_seeds}, start_seed={args.start_seed}, "
        f"rounds={args.rounds}, noise={args.noise}, bracket_type={args.bracket_type}"
    )

    runs: List[SeedRunResult] = []
    for offset in range(args.num_seeds):
        seed = args.start_seed + offset
        run = run_bracket_for_seed(
            seed=seed,
            rounds=args.rounds,
            noise=args.noise,
            bracket_type=args.bracket_type,
        )
        runs.append(run)
        if args.detail_level != "summary":
            print_run_details(run, args.detail_level)

    summary = summarize_runs(runs, GAME_TYPE)
    print_summary(summary)

    json_out = outfile.replace(".txt", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {json_out}")


if __name__ == "__main__":
    main()
