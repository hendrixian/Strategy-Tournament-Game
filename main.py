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

from bracket import BracketTournament
from bracket_visualization import BracketVisualizer


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
    
    # Plot 2: Cooperation network
    analyzer.plot_cooperation_network(results['match_results'], axes[0, 1])
    
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

def run_bracket_demo():
    """Run a bracket tournament demonstration"""
    print("=" * 60)
    print("🏆 BRACKET TOURNAMENT DEMO")
    print("=" * 60)
    
    # Get strategies
    strategies = get_all_strategies()
    print(f"\nAvailable Strategies ({len(strategies)}):")
    for i, s in enumerate(strategies, 1):
        print(f"{i:2}. {s.name:20}")
    
    # Create payoff matrix
    payoff = PayoffMatrix(PayoffMatrix.PRISONERS_DILEMMA, T=5, R=3, P=1, S=0)
    print(f"\nGame: {payoff}")
    
    # Run single elimination
    print("\n" + "=" * 60)
    print("SINGLE ELIMINATION TOURNAMENT")
    print("=" * 60)
    
    # Run single elimination bracket
    single_bracket = BracketTournament(
        strategies, 
        payoff, 
        rounds_per_match=10, 
        noise=0.05,
        bracket_type="single"
    )
    
    single_results = single_bracket.run_tournament()
    single_summary = BracketVisualizer.create_bracket_summary_table(single_results)
    
    print(f"\n🏆 Champion: {single_summary['champion']}")
    print(f"🥈 Runner-up: {single_summary['runner_up']}")
    if single_summary.get('third_place'):
        print(f"🥉 3rd Place: {single_summary['third_place']}")
    
    print(f"\n📊 Tournament Statistics:")
    print(f"Total Matches: {single_summary['total_matches']}")
    print(f"Total Strategies: {single_summary['total_strategies']}")
    
    # Display strategy stats
    print("\n" + "-" * 60)
    print("STRATEGY PERFORMANCE")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Wins':<6} {'Losses':<7} {'Win Rate':<10} {'Avg Score':<10}")
    print("-" * 60)
    
    for name, stats in single_summary['strategy_stats'].items():
        print(f"{name:<20} {stats['wins']:<6} {stats['losses']:<7} "
              f"{stats['win_rate']:<10.1%} {stats['avg_score']:<10.2f}")
    
    # Visualize bracket
    print("\n📈 Generating bracket visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot bracket
    single_data = single_bracket.get_bracket_visualization_data(single_results)
    BracketVisualizer.plot_single_elimination_bracket(single_data, axes[0])
    
    # Plot win distribution
    strategies_list = list(single_summary['strategy_stats'].keys())
    wins = [single_summary['strategy_stats'][s]['wins'] for s in strategies_list]
    
    bars = axes[1].barh(strategies_list, wins, color='steelblue')
    axes[1].set_xlabel('Number of Wins')
    axes[1].set_title('Bracket Tournament Wins')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width, bar.get_y() + bar.get_height()/2,
                    f' {width}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Run double elimination if user wants
    print("\n" + "=" * 60)
    print("DOUBLE ELIMINATION TOURNAMENT")
    print("=" * 60)
    
    response = input("\nRun double elimination tournament? (y/n): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        double_bracket = BracketTournament(
            strategies, 
            payoff, 
            rounds_per_match=10, 
            noise=0.05,
            bracket_type="double"
        )
        
        double_results = double_bracket.run_tournament()
        double_summary = BracketVisualizer.create_bracket_summary_table(double_results)
        
        print(f"\n🏆 Champion: {double_summary['champion']}")
        print(f"🥈 Runner-up: {double_summary['runner_up']}")
        
        print(f"\n📊 Tournament Statistics:")
        print(f"Total Matches: {double_summary['total_matches']}")
        
        # Visualize double elimination bracket
        print("\n📈 Generating double elimination bracket visualization...")
        fig, ax = plt.subplots(figsize=(14, 10))
        double_data = double_bracket.get_bracket_visualization_data(double_results)
        BracketVisualizer.plot_double_elimination_bracket(double_data, ax)
        plt.tight_layout()
        plt.show()
    
    print("\n🎉 Bracket tournament simulation complete!")

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

    print("🎮 STRATEGY TOURNAMENT GAME")
    print("=" * 40)
    print("Select demo type:")
    print("1. Standard Round-Robin Tournament")
    print("2. Bracket Tournament (Single & Double Elimination)")
    print("3. Both (Run all demos)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    try:
        if choice == "1":
            run_tournament_demo()
        elif choice == "2":
            run_bracket_demo()
        elif choice == "3":
            run_tournament_demo()
            print("\n" + "=" * 60)
            print("NOW RUNNING BRACKET TOURNAMENT DEMO")
            print("=" * 60)
            run_bracket_demo()
        else:
            print("Invalid choice. Running standard tournament.")
            run_tournament_demo()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()