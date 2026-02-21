"""
Run evolutionary dynamics experiments for report figures/tables.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from evolution import EvolutionaryDynamics
from payoff import PayoffMatrix
from strategies import get_all_strategies


@dataclass(frozen=True)
class GameSpec:
    name: str
    payoff_type: str


GAMES: List[GameSpec] = [
    GameSpec("Prisoner's Dilemma", PayoffMatrix.PRISONERS_DILEMMA),
    GameSpec("Snowdrift Game", PayoffMatrix.SNOWDRIFT),
    GameSpec("Stag Hunt", PayoffMatrix.STAG_HUNT),
    GameSpec("Matching Pennies", PayoffMatrix.MATCHING_PENNIES),
]

ESS_GAMES = {"Prisoner's Dilemma", "Stag Hunt"}


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_replicator(
    payoff_type: str,
    generations: int,
    mutation_rate: float,
    seed: int,
) -> Tuple[List[str], np.ndarray, List[float]]:
    set_seeds(seed)
    strategies = get_all_strategies()
    payoff = PayoffMatrix(payoff_type)
    evo = EvolutionaryDynamics(strategies, payoff)
    initial_pop = np.ones(len(strategies)) / len(strategies)
    pop_history, avg_fitness = evo.replicator_dynamics(
        initial_pop, generations=generations, mutation_rate=mutation_rate
    )
    names = [s.name for s in strategies]
    return names, pop_history, avg_fitness


def plot_trajectory(
    game_name: str,
    strategy_names: List[str],
    pop_history: np.ndarray,
    generations: int,
    mutation_rate: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(generations)

    for idx, name in enumerate(strategy_names):
        ax.plot(x, pop_history[:, idx], label=name, linewidth=1.6)

    ax.set_title(
        f"{game_name} | Replicator Dynamics\n"
        f"Generations: {generations} | Mutation: {mutation_rate}"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Share")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_final_shares_table(
    out_path: str,
    rows: List[Dict[str, object]],
) -> None:
    fieldnames = ["game", "strategy", "final_share", "final_avg_fitness"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mutation_compare_table(
    out_path: str,
    rows: List[Dict[str, object]],
) -> None:
    fieldnames = [
        "game",
        "strategy",
        "final_share_mu0",
        "final_share_mu001",
        "delta_mu001_minus_mu0",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_ess_table(out_path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["game", "strategy", "is_ess", "fitness_against_self"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_invasion_table(out_path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "game",
        "resident",
        "mutant",
        "initial_mutant_share",
        "final_mutant_share",
        "invaded",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_ess_and_invasion(
    game_name: str,
    payoff_type: str,
    generations: int,
    seed: int,
    mutant_share: float = 0.05,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    set_seeds(seed)
    strategies = get_all_strategies()
    payoff = PayoffMatrix(payoff_type)
    evo = EvolutionaryDynamics(strategies, payoff)

    ess_results = evo.find_all_ess()
    ess_rows: List[Dict[str, object]] = []
    for res in ess_results:
        ess_rows.append(
            {
                "game": game_name,
                "strategy": res["strategy"],
                "is_ess": bool(res["is_ess"]),
                "fitness_against_self": float(res["fitness_against_self"]),
            }
        )

    invasion_rows: List[Dict[str, object]] = []
    name_to_idx = {s.name: idx for idx, s in enumerate(strategies)}
    ess_candidates = [r["strategy"] for r in ess_results if r["is_ess"]]

    for resident in ess_candidates:
        resident_idx = name_to_idx[resident]
        for mutant in name_to_idx.keys():
            if mutant == resident:
                continue
            mutant_idx = name_to_idx[mutant]
            initial_pop = np.zeros(len(strategies))
            initial_pop[resident_idx] = 1.0 - mutant_share
            initial_pop[mutant_idx] = mutant_share

            set_seeds(seed)
            pop_history, _ = evo.replicator_dynamics(
                initial_pop,
                generations=generations,
                mutation_rate=0.0,
            )
            final_mutant_share = float(pop_history[-1, mutant_idx])
            invasion_rows.append(
                {
                    "game": game_name,
                    "resident": resident,
                    "mutant": mutant,
                    "initial_mutant_share": mutant_share,
                    "final_mutant_share": final_mutant_share,
                    "invaded": final_mutant_share > mutant_share,
                }
            )

    return ess_rows, invasion_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evolutionary dynamics experiments for report figures/tables."
    )
    parser.add_argument("--generations", type=int, default=200, help="Generations")
    parser.add_argument("--mutation", type=float, default=0.01, help="Mutation rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out-dir", type=str, default="results/evolution", help="Output directory"
    )
    parser.add_argument(
        "--compare-mutation",
        action="store_true",
        help="Also compare mu=0 vs mu=0.01 final shares",
    )
    parser.add_argument(
        "--ess",
        action="store_true",
        help="Run ESS + invasion tests for Prisoner's Dilemma and Stag Hunt",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    baseline_rows: List[Dict[str, object]] = []
    compare_rows: List[Dict[str, object]] = []
    ess_rows: List[Dict[str, object]] = []
    invasion_rows: List[Dict[str, object]] = []

    for game in GAMES:
        names, pop_history, avg_fitness = run_replicator(
            game.payoff_type,
            generations=args.generations,
            mutation_rate=args.mutation,
            seed=args.seed,
        )
        final_pop = pop_history[-1]
        final_avg_fitness = float(avg_fitness[-1]) if avg_fitness else float("nan")

        for idx, name in enumerate(names):
            baseline_rows.append(
                {
                    "game": game.name,
                    "strategy": name,
                    "final_share": float(final_pop[idx]),
                    "final_avg_fitness": final_avg_fitness,
                }
            )

        plot_path = os.path.join(
            args.out_dir,
            f"trajectory_{slugify(game.name)}_mu{args.mutation}_gen{args.generations}.png",
        )
        plot_trajectory(
            game.name,
            names,
            pop_history,
            args.generations,
            args.mutation,
            plot_path,
        )

        if args.compare_mutation:
            names_zero, pop_zero, _ = run_replicator(
                game.payoff_type,
                generations=args.generations,
                mutation_rate=0.0,
                seed=args.seed,
            )
            final_zero = pop_zero[-1]
            for idx, name in enumerate(names_zero):
                compare_rows.append(
                    {
                        "game": game.name,
                        "strategy": name,
                        "final_share_mu0": float(final_zero[idx]),
                        "final_share_mu001": float(final_pop[idx]),
                        "delta_mu001_minus_mu0": float(final_pop[idx] - final_zero[idx]),
                    }
                )

        if args.ess and game.name in ESS_GAMES:
            game_ess_rows, game_invasion_rows = run_ess_and_invasion(
                game.name,
                game.payoff_type,
                generations=args.generations,
                seed=args.seed,
            )
            ess_rows.extend(game_ess_rows)
            invasion_rows.extend(game_invasion_rows)

    baseline_path = os.path.join(
        args.out_dir, f"final_shares_mu{args.mutation}_gen{args.generations}.csv"
    )
    write_final_shares_table(baseline_path, baseline_rows)

    if args.compare_mutation:
        compare_path = os.path.join(
            args.out_dir,
            f"final_shares_mu0_vs_mu{args.mutation}_gen{args.generations}.csv",
        )
        write_mutation_compare_table(compare_path, compare_rows)

    if args.ess:
        ess_path = os.path.join(
            args.out_dir, f"ess_results_gen{args.generations}.csv"
        )
        write_ess_table(ess_path, ess_rows)

        invasion_path = os.path.join(
            args.out_dir, f"invasion_tests_gen{args.generations}.csv"
        )
        write_invasion_table(invasion_path, invasion_rows)

    print("Evolution experiments complete.")
    print(f"Output directory: {args.out_dir}")
    print(f"Baseline final shares: {baseline_path}")
    if args.compare_mutation:
        print(f"Mutation comparison: {compare_path}")
    if args.ess:
        print(f"ESS results: {ess_path}")
        print(f"Invasion tests: {invasion_path}")


if __name__ == "__main__":
    main()
