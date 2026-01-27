"""
Main script to run the tournament simulation.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from strategies import get_all_strategies, get_strategy_by_name
from payoff import PayoffMatrix
from tournament import Tournament
from evolution import EvolutionaryDynamics
from analysis import TournamentAnalyzer, analyze_nash_equilibrium

def run_tournament_demo():
    """Run a demonstration tournament"""
    print("=" * 60)
    print("STRATEGY TOURNAMENT GAME - GAME THEORY SIMULATOR")
    print("=" * 60)
    
    # Get strategies
    strategies = get_all_strategies()
    print(f"\nAvailable Strategies ({len(strategies)}):")
    for i, s in enumerate(strategies, 1):
        print(f"{i:2}. {s.name:20} - {s.description}")
    
    # Create payoff matrix (Prisoner's Dilemma by default)
    payoff = PayoffMatrix(PayoffMatrix.PRISONERS_DILEMMA, T=5, R=3, P=1, S=0)
    print(f"\nGame: {payoff}")
    
    # Analyze Nash equilibrium
    print("\n" + "=" * 60)
    print("GAME ANALYSIS")
    print("=" * 60)
    nash_table = analyze_nash_equilibrium(payoff)
    print(nash_table.to_string(index=False))
    
    # Create and run tournament
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    tournament = Tournament(strategies, payoff, rounds_per_match=10, noise=0.05)
    results = tournament.run_round_robin()
    
    # Add cooperation rates to results
    results['cooperation_rates'] = tournament.get_cooperation_rates()
    
    # Create analyzer and show results
    analyzer = TournamentAnalyzer(results)
    summary = analyzer.create_summary_table()
    print(summary.to_string(index=False))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score distribution
    analyzer.plot_score_distribution(axes[0, 0])
    
    # Plot 2: Cooperation heatmap
    analyzer.plot_cooperation_heatmap(results['match_results'], axes[0, 1])
    
    # Plot 3: Payoff matrix
    analyzer.plot_payoff_matrix(payoff, axes[1, 0])
    
    # Plot 4: Example match history
    if results['match_results']:
        # Use first match as example
        example_match = results['match_results'][0]
        analyzer.plot_match_history(example_match['history'], axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Evolutionary dynamics
    print("\n" + "=" * 60)
    print("EVOLUTIONARY DYNAMICS")
    print("=" * 60)
    
    # Initialize evolutionary dynamics
    evo = EvolutionaryDynamics(strategies, payoff)
    
    # Check for ESS
    ess_results = evo.find_all_ess()
    print("\nEvolutionarily Stable Strategies (ESS) Analysis:")
    for result in ess_results:
        status = "✓ ESS" if result['is_ess'] else "✗ Not ESS"
        print(f"\n{result['strategy']}: {status}")
        if result['conditions']:
            for condition in result['conditions'][:2]:  # Show first 2 conditions
                print(f"  - {condition}")
    
    # Run replicator dynamics
    print("\nSimulating replicator dynamics...")
    initial_pop = np.ones(len(strategies))  # Equal initial frequencies
    pop_history, fitness_history = evo.replicator_dynamics(
        initial_pop, generations=100, mutation_rate=0.01
    )
    
    # Plot evolutionary dynamics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Population frequencies
    for i, s in enumerate(strategies):
        ax1.plot(pop_history[:, i], label=s.name, linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population Frequency')
    ax1.set_title('Evolutionary Dynamics (Replicator Equation)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Average fitness
    ax2.plot(fitness_history, 'k-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Population Fitness')
    ax2.set_title('Evolution of Average Fitness')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTournament simulation complete!")

def main():
    """Main entry point"""
    try:
        run_tournament_demo()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()