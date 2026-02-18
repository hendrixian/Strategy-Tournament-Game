"""Run repeated tournament simulations across many seeds with optional evolution summary."""

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from evolution import EvolutionaryDynamics
from payoff import PayoffMatrix
from strategies import get_all_strategies
from tournament import Tournament


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
    winner: str
    scores: Dict[str, float]
    ranks: Dict[str, int]
    match_results: List[Dict[str, Any]]
    evolution_final_population: Dict[str, float]
    evolution_final_avg_fitness: float


def run_tournament_for_seed(
    seed: int,
    rounds: int,
    noise: float,
    generations: int,
    mutation_rate: float,
) -> SeedRunResult:
    """Run one tournament and one evolutionary simulation under a single seed."""
    random.seed(seed)
    np.random.seed(seed)

    strategies = get_all_strategies()
    payoff = PayoffMatrix(PayoffMatrix.PRISONERS_DILEMMA)

    tournament = Tournament(strategies, payoff, rounds_per_match=rounds, noise=noise)
    results = tournament.run_round_robin()
    rankings = results["rankings"]

    ranks = {name: idx + 1 for idx, (name, _) in enumerate(rankings)}
    winner = rankings[0][0]

    evo = EvolutionaryDynamics(strategies, payoff)
    initial_pop = np.ones(len(strategies)) / len(strategies)
    pop_history, fitness_history = evo.replicator_dynamics(
        initial_pop, generations=generations, mutation_rate=mutation_rate
    )
    final_pop = pop_history[-1]
    evolution_final_population = {
        strategy.name: float(final_pop[idx]) for idx, strategy in enumerate(strategies)
    }
    evolution_final_avg_fitness = float(fitness_history[-1]) if fitness_history else float("nan")

    return SeedRunResult(
        seed=seed,
        winner=winner,
        scores=results["scores"],
        ranks=ranks,
        match_results=results["match_results"],
        evolution_final_population=evolution_final_population,
        evolution_final_avg_fitness=evolution_final_avg_fitness,
    )


def summarize_runs(runs: List[SeedRunResult]) -> Dict[str, Any]:
    """Compute aggregate statistics across all runs."""
    strategy_names = sorted(runs[0].scores.keys())

    win_counts = {name: 0 for name in strategy_names}
    score_samples = {name: [] for name in strategy_names}
    rank_samples = {name: [] for name in strategy_names}
    evo_final_pop_samples = {name: [] for name in strategy_names}
    evo_top_counts = {name: 0 for name in strategy_names}
    evo_avg_fitness_samples = []

    for run in runs:
        win_counts[run.winner] += 1
        for name in strategy_names:
            score_samples[name].append(run.scores[name])
            rank_samples[name].append(run.ranks[name])
            evo_final_pop_samples[name].append(run.evolution_final_population[name])

        evo_winner = max(run.evolution_final_population.items(), key=lambda item: item[1])[0]
        evo_top_counts[evo_winner] += 1
        evo_avg_fitness_samples.append(run.evolution_final_avg_fitness)

    total_runs = len(runs)

    summary_rows = []
    for name in strategy_names:
        summary_rows.append(
            {
                "strategy": name,
                "win_count": win_counts[name],
                "win_rate": win_counts[name] / total_runs,
                "avg_rank": statistics.mean(rank_samples[name]),
                "avg_score": statistics.mean(score_samples[name]),
                "std_score": statistics.pstdev(score_samples[name]),
            }
        )
    summary_rows.sort(key=lambda row: (-row["win_count"], row["avg_rank"], -row["avg_score"]))

    evolution_summary_rows = []
    for name in strategy_names:
        evolution_summary_rows.append(
            {
                "strategy": name,
                "top_final_pop_count": evo_top_counts[name],
                "top_final_pop_rate": evo_top_counts[name] / total_runs,
                "avg_final_population": statistics.mean(evo_final_pop_samples[name]),
                "std_final_population": statistics.pstdev(evo_final_pop_samples[name]),
            }
        )
    evolution_summary_rows.sort(
        key=lambda row: (-row["top_final_pop_count"], -row["avg_final_population"])
    )

    return {
        "total_runs": total_runs,
        "summary_rows": summary_rows,
        "winners_by_seed": [(run.seed, run.winner) for run in runs],
        "rank_samples": rank_samples,
        "evolution_summary_rows": evolution_summary_rows,
        "evolution_winners_by_seed": [
            (run.seed, max(run.evolution_final_population.items(), key=lambda item: item[1])[0])
            for run in runs
        ],
        "evolution_avg_fitness": {
            "mean": statistics.mean(evo_avg_fitness_samples),
            "std": statistics.pstdev(evo_avg_fitness_samples),
        },
    }


def print_run_details(run: SeedRunResult, detail_level: str) -> None:
    """Print detailed information for one run."""
    print("=" * 90)
    print(f"Run seed={run.seed}")
    print("Step 1/4: Random generators seeded")
    print("Step 2/4: Tournament completed")

    if detail_level in {"match", "round"}:
        print("Step 3/4: Match-by-match outcomes")
        for idx, match in enumerate(run.match_results, start=1):
            s1 = match["strategy1"]
            s2 = match["strategy2"]
            print(
                f"  Match {idx:02d}: {s1} vs {s2} | "
                f"scores=({match['score1']:.2f}, {match['score2']:.2f})"
            )

            if detail_level == "round":
                for round_idx, (m1, m2, p1, p2) in enumerate(match["history"], start=1):
                    print(
                        f"    Round {round_idx:02d}: {s1}:{m1} {s2}:{m2} "
                        f"-> payoff=({p1:.2f}, {p2:.2f})"
                    )

    print("Step 4/4: Final rankings")
    sorted_rankings = sorted(run.ranks.items(), key=lambda item: item[1])
    for name, rank in sorted_rankings:
        print(f"  #{rank}: {name:20s} score={run.scores[name]:.4f}")
    print(f"Winner: {run.winner}")


def print_summary(summary: Dict[str, Any]) -> None:
    """Print aggregate results over all seeds."""
    print("\n" + "#" * 90)
    print("Aggregate multi-seed summary")
    print("#" * 90)
    print(f"Total runs: {summary['total_runs']}")

    print("\nWinner by seed:")
    for seed, winner in summary["winners_by_seed"]:
        print(f"  seed={seed:4d} -> winner={winner}")

    print("\nPer-strategy aggregate metrics:")
    print("  (win count / win rate / average rank / average score +/- std)")
    for row in summary["summary_rows"]:
        print(
            f"  {row['strategy']:20s} "
            f"wins={row['win_count']:3d} "
            f"win_rate={row['win_rate']:.2%} "
            f"avg_rank={row['avg_rank']:.3f} "
            f"avg_score={row['avg_score']:.4f} +/- {row['std_score']:.4f}"
        )

    print("\nEvolution winner by seed (highest final population share):")
    for seed, winner in summary["evolution_winners_by_seed"]:
        print(f"  seed={seed:4d} -> evolution_winner={winner}")

    print("\nPer-strategy evolution aggregate metrics:")
    print("  (top final population count / rate / average final population +/- std)")
    for row in summary["evolution_summary_rows"]:
        print(
            f"  {row['strategy']:20s} "
            f"top_count={row['top_final_pop_count']:3d} "
            f"top_rate={row['top_final_pop_rate']:.2%} "
            f"avg_final_pop={row['avg_final_population']:.4f} "
            f"+/- {row['std_final_population']:.4f}"
        )
    print(
        "Evolution average fitness (final generation): "
        f"{summary['evolution_avg_fitness']['mean']:.4f} "
        f"+/- {summary['evolution_avg_fitness']['std']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated round-robin tournaments over many seeds and report rankings."
    )
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds/runs")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per match")
    parser.add_argument("--noise", type=float, default=0.20, help="Noise level [0,1]")
    parser.add_argument("--generations", type=int, default=50, help="Evolution generations")
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Evolution mutation rate [0,1]",
    )
    parser.add_argument(
        "--detail-level",
        choices=["summary", "match", "round"],
        default="match",
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
    if args.generations <= 0:
        raise ValueError("--generations must be > 0")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("--mutation-rate must be between 0 and 1")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        results_dir,
        f"tournament_{args.num_seeds}s_{args.rounds}r_noise{args.noise}_"
        f"gen{args.generations}_mut{args.mutation_rate}_{timestamp}.txt",
    )
    sys.stdout = Tee(outfile)
    print(f"Output saved to: {outfile}")

    print("Running multi-seed tournament + evolution experiment")
    print(
        "Configuration: "
        f"num_seeds={args.num_seeds}, start_seed={args.start_seed}, "
        f"rounds={args.rounds}, noise={args.noise}, "
        f"generations={args.generations}, mutation_rate={args.mutation_rate}"
    )

    runs: List[SeedRunResult] = []
    for offset in range(args.num_seeds):
        seed = args.start_seed + offset
        run = run_tournament_for_seed(
            seed=seed,
            rounds=args.rounds,
            noise=args.noise,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
        )
        runs.append(run)
        if args.detail_level != "summary":
            print_run_details(run, args.detail_level)

    summary = summarize_runs(runs)
    print_summary(summary)

    json_out = outfile.replace(".txt", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"JSON summary saved to: {json_out}")


if __name__ == "__main__":
    main()
