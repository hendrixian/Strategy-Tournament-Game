"""
Streamlit web application for the Strategy Tournament Game.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from strategies import get_all_strategies, get_strategy_by_name
from payoff import PayoffMatrix
from tournament import Tournament
from evolution import EvolutionaryDynamics
from analysis import TournamentAnalyzer, analyze_nash_equilibrium, compute_strategy_metrics, plot_strategy_radar, normalize_metrics

from bracket import BracketTournament
from bracket_visualization import BracketVisualizer
from sklearn.tree import export_text

#added by thu for decision tree
from decision_tree_analysis import (
    prepare_decision_tree_data,
    train_decision_tree,
    plot_decision_tree
)


# Page configuration
st.set_page_config(
    page_title="Strategy Tournament Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .radar-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'prelim_rankings' not in st.session_state:
        st.session_state.prelim_rankings = []
    if 'radar_metrics' not in st.session_state:
        st.session_state.radar_metrics = None
    if 'tournament_results' not in st.session_state:
        st.session_state.tournament_results = None
    if 'bracket_results' not in st.session_state:
        st.session_state.bracket_results = None

def main():
    """Main Streamlit app"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎮 Strategy Tournament Game</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">An Interactive Game Theory Simulator</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Tournament Configuration")
        
        # Game selection
        game_type = st.selectbox(
            "Game Type",
            [PayoffMatrix.PRISONERS_DILEMMA, PayoffMatrix.SNOWDRIFT, 
             PayoffMatrix.STAG_HUNT, PayoffMatrix.MATCHING_PENNIES],
            index=0
        )
        
        # Custom payoffs for symmetric games
        if game_type != PayoffMatrix.MATCHING_PENNIES:
            col1, col2 = st.columns(2)
            with col1:
                T = st.number_input("T (Temptation)", value=5.0, step=0.5)
                R = st.number_input("R (Reward)", value=3.0, step=0.5)
            with col2:
                P = st.number_input("P (Punishment)", value=1.0, step=0.5)
                S = st.number_input("S (Sucker)", value=0.0, step=0.5)
            payoff_kwargs = {'T': T, 'R': R, 'P': P, 'S': S}
        else:
            payoff_kwargs = {'win': 1, 'lose': -1}
        
        # Tournament parameters
        st.subheader("Tournament Settings")
        rounds_per_match = st.slider("Rounds per Match", 5, 50, 10)
        noise = st.slider("Noise Level", 0.0, 0.2, 0.05, step=0.01)
        
        # Strategy selection
        st.subheader("Strategy Selection")
        all_strategies = get_all_strategies()
        selected_strategies = []
        
        for strategy in all_strategies:
            if st.checkbox(strategy.name, value=True, help=strategy.description):
                selected_strategies.append(strategy)
        
        if not selected_strategies:
            st.warning("Please select at least one strategy!")
            st.stop()
        
        # Evolutionary simulation parameters
        st.subheader("Evolutionary Simulation")
        evo_generations = st.slider("Generations", 10, 200, 50)
        mutation_rate = st.slider("Mutation Rate", 0.0, 0.1, 0.01, step=0.005)
    
    # Main content area - Added Strategy Radar tab
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📊 Tournament", "📈 Evolution", "🎮 Single Match", "📚 Theory", "🏆 Bracket", "🎯 Strategy Radar", "🌳 Decision Tree"]
    )
    
    # Initialize payoff matrix
    try:
        payoff = PayoffMatrix(game_type, **payoff_kwargs)
    except ValueError as e:
        st.error(f"Invalid payoff parameters: {e}")
        st.stop()
    
    with tab1:
        st.markdown('<h2 class="sub-header">Tournament Results</h2>', unsafe_allow_html=True)
        
        if st.button("🏆 Run Tournament", type="primary", width='stretch'):
            with st.spinner("Running tournament..."):
                # Create and run tournament
                tournament = Tournament(selected_strategies, payoff, 
                                      rounds_per_match=rounds_per_match, 
                                      noise=noise)
                results = tournament.run_round_robin()
                results['cooperation_rates'] = tournament.get_cooperation_rates()
                
                # Display results
                analyzer = TournamentAnalyzer(results)
                
                # Summary table
                st.subheader("Tournament Rankings")
                summary = analyzer.create_summary_table()
                st.dataframe(summary, width='stretch', hide_index=True)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Strategy Performance")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    analyzer.plot_score_distribution(ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.subheader("Cooperation Network")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    analyzer.plot_cooperation_network(results['match_results'], ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Store results for radar analysis
                st.session_state.tournament_results = results
                st.session_state.tournament = tournament
                
                # Match history example
                st.subheader("Example Match Analysis")
                if results['match_results']:
                    # Let user select a match
                    match_options = [f"{m['strategy1']} vs {m['strategy2']}" 
                                   for m in results['match_results']]
                    selected_match = st.selectbox("Select a match to analyze:", match_options)
                    
                    match_idx = match_options.index(selected_match)
                    match_data = results['match_results'][match_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Match summary
                        st.metric(f"{match_data['strategy1']} Score", 
                                 f"{match_data['score1']:.2f}")
                        st.metric(f"{match_data['strategy2']} Score", 
                                 f"{match_data['score2']:.2f}")
                        
                        # Cooperation rates in this match
                        history = match_data.get('history', [])
                        if history:
                            coop1 = sum(1 for h in history if h[0] == 'C') / len(history)
                            coop2 = sum(1 for h in history if h[1] == 'C') / len(history)
                            st.metric(f"{match_data['strategy1']} Cooperation", 
                                     f"{coop1:.1%}")
                            st.metric(f"{match_data['strategy2']} Cooperation", 
                                     f"{coop2:.1%}")
                    
                    with col2:
                        # Plot match history
                        if history:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            analyzer.plot_match_history(history, ax)
                            st.pyplot(fig)
                            plt.close(fig)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Evolutionary Dynamics</h2>', unsafe_allow_html=True)
        
        if st.button("🌱 Run Evolutionary Simulation", type="primary", width='stretch'):
            with st.spinner("Running evolutionary simulation..."):
                # Initialize evolutionary dynamics
                evo = EvolutionaryDynamics(selected_strategies, payoff)
                
                # Run replicator dynamics
                initial_pop = np.ones(len(selected_strategies)) / len(selected_strategies)
                pop_history, fitness_history = evo.replicator_dynamics(
                    initial_pop, generations=evo_generations, mutation_rate=mutation_rate
                )
                
                # Display ESS analysis
                st.subheader("Evolutionarily Stable Strategies (ESS)")
                ess_results = evo.find_all_ess()
                
                ess_data = []
                for result in ess_results:
                    ess_data.append({
                        'Strategy': result['strategy'],
                        'ESS Status': '✓ Stable' if result['is_ess'] else '✗ Not Stable',
                        'Fitness Against Self': f"{result['fitness_against_self']:.3f}"
                    })
                
                ess_df = pd.DataFrame(ess_data)
                st.dataframe(ess_df, width='stretch', hide_index=True)
                
                # Plot evolutionary dynamics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Population Dynamics")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    strategy_names = [s.name for s in selected_strategies]
                    for i, name in enumerate(strategy_names):
                        ax.plot(pop_history[:, i], label=name, linewidth=2)
                    
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Population Frequency')
                    ax.set_title('Evolutionary Dynamics (Replicator Equation)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.subheader("Fitness Evolution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.plot(fitness_history, 'k-', linewidth=2)
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Average Population Fitness')
                    ax.set_title('Evolution of Average Fitness')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Final population distribution
                st.subheader("Final Population Distribution")
                final_pop = pop_history[-1, :]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(strategy_names, final_pop, color='steelblue')
                ax.set_xlabel('Population Frequency')
                ax.set_title('Final Population Distribution')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f' {width:.3f}', va='center')
                
                st.pyplot(fig)
                plt.close(fig)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Single Match Simulator</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy1_name = st.selectbox(
                "Player 1 Strategy",
                [s.name for s in selected_strategies],
                index=0
            )
            strategy1 = get_strategy_by_name(strategy1_name)
        
        with col2:
            strategy2_name = st.selectbox(
                "Player 2 Strategy",
                [s.name for s in selected_strategies],
                index=min(1, len(selected_strategies)-1)
            )
            strategy2 = get_strategy_by_name(strategy2_name)
        
        match_rounds = st.slider("Number of Rounds", 1, 50, 10)
        
        if st.button("⚔️ Simulate Match", type="primary"):
            # Simulate match
            tournament = Tournament([strategy1, strategy2], payoff, 
                                  rounds_per_match=match_rounds, noise=noise)
            score1, score2, history = tournament.play_match(strategy1, strategy2)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{strategy1.name} Score", f"{score1:.2f}")
                coop_rate1 = sum(1 for h in history if h[0] == 'C') / len(history)
                st.metric(f"{strategy1.name} Cooperation", f"{coop_rate1:.1%}")
            
            with col2:
                st.metric(f"{strategy2.name} Score", f"{score2:.2f}")
                coop_rate2 = sum(1 for h in history if h[1] == 'C') / len(history)
                st.metric(f"{strategy2.name} Cooperation", f"{coop_rate2:.1%}")
            
            # Show detailed moves
            st.subheader("Round-by-Round Details")
            moves_data = []
            for i, (m1, m2, p1, p2) in enumerate(history, 1):
                moves_data.append({
                    'Round': i,
                    f'{strategy1.name}': m1,
                    f'{strategy2.name}': m2,
                    f'{strategy1.name} Payoff': p1,
                    f'{strategy2.name} Payoff': p2
                })
            
            moves_df = pd.DataFrame(moves_data)
            st.dataframe(moves_df, width='stretch', hide_index=True)
            
            # Plot match history
            fig, ax = plt.subplots(figsize=(12, 4))
            analyzer = TournamentAnalyzer({})
            analyzer.plot_match_history(history, ax)
            st.pyplot(fig)
            plt.close(fig)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Game Theory Concepts</h2>', unsafe_allow_html=True)
        
        # Game analysis
        st.subheader("Game Analysis")
        nash_table = analyze_nash_equilibrium(payoff)
        st.dataframe(nash_table, width='stretch', hide_index=True)
        
        # Payoff matrix visualization
        st.subheader("Payoff Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        analyzer = TournamentAnalyzer({})
        analyzer.plot_payoff_matrix(payoff, ax)
        st.pyplot(fig)
        plt.close(fig)
        
        # Game theory explanations
        st.subheader("Key Concepts")
        
        with st.expander("Prisoner's Dilemma"):
            st.markdown("""
            The Prisoner's Dilemma is a classic game theory scenario where:
            - Two players can either **Cooperate (C)** or **Defect (D)**
            - If both cooperate: Both get **R** (Reward)
            - If both defect: Both get **P** (Punishment)
            - If one cooperates and one defects: Cooperator gets **S** (Sucker), Defector gets **T** (Temptation)
            
            The dilemma arises because:
            1. **D** dominates **C** for each player individually
            2. But (D,D) gives worse outcomes than (C,C)
            
            Conditions: T > R > P > S and 2R > T + S
            """)
        
        with st.expander("Nash Equilibrium"):
            st.markdown("""
            A Nash Equilibrium is a situation where no player can improve their payoff
            by unilaterally changing their strategy, given what the other players are doing.
            
            In the Prisoner's Dilemma:
            - **(D,D)** is the only Nash Equilibrium in pure strategies
            - It's also a **dominant strategy equilibrium**
            
            However, in repeated games, cooperation can emerge as an equilibrium!
            """)
        
        with st.expander("Evolutionarily Stable Strategy (ESS)"):
            st.markdown("""
            An Evolutionarily Stable Strategy (ESS) is a strategy that:
            1. Cannot be invaded by any alternative strategy that is initially rare
            2. If everyone plays the ESS, no mutant strategy can do better
            
            Mathematically, a strategy **s** is ESS if for all **t ≠ s**:
            - Either E(s,s) > E(t,s), or
            - E(s,s) = E(t,s) and E(s,t) > E(t,t)
            
            ESS is a refinement of Nash Equilibrium for evolutionary games.
            """)
        
        with st.expander("Common Strategies"):
            st.markdown("""
            **Always Cooperate (AC)**: Always chooses C  
            **Always Defect (AD)**: Always chooses D  
            **Tit for Tat (TFT)**: Start with C, then copy opponent's last move  
            **Tit for Two Tats**: Defect only after two consecutive D's from opponent  
            **Grudger**: Cooperate until opponent defects, then always defect  
            **Pavlov (Win-Stay, Lose-Shift)**: Repeat if payoff was good, switch if payoff was bad  
            **Random**: Randomly choose between C and D
            """)

    with tab5:
        st.markdown('<h2 class="sub-header">Tournament Bracket</h2>', unsafe_allow_html=True)
        
        # Information box
        st.markdown("""
        <div class="info-box">
        <strong>🏆 Bracket Tournament Features:</strong><br>
        • <strong>Single Elimination:</strong> One loss and you're out!<br>
        • <strong>Double Elimination:</strong> Get a second chance in the loser's bracket<br>
        • <strong>Seeding:</strong> Random or performance-based matchups<br>
        • <strong>Visual Brackets:</strong> See the tournament tree unfold<br>
        • <strong>Detailed Stats:</strong> Win rates, match histories, and more
        </div>
        """, unsafe_allow_html=True)
        
        # Bracket configuration
        col1, col2 = st.columns(2)
        
        with col1:
            bracket_type = st.selectbox(
                "Bracket Type",
                ["Single Elimination", "Double Elimination"],
                index=0,
                key="bracket_type"
            )
            
            seeding_method = st.radio(
                "Seeding Method",
                ["Random", "Ranked (based on round-robin performance)"],
                index=0,
                key="seeding_method"
            )
            
            # Option to run round-robin first for ranking
            if seeding_method == "Ranked (based on round-robin performance)":
                if st.button("📊 Run Preliminary Round-Robin", key="run_prelim"):
                    with st.spinner("Running preliminary round-robin tournament..."):
                        prelim_tournament = Tournament(selected_strategies, payoff, 
                                                    rounds_per_match=rounds_per_match, 
                                                    noise=noise)
                        prelim_results = prelim_tournament.run_round_robin()
                        
                        # Store rankings for later use
                        st.session_state.prelim_rankings = [
                            name for name, score in prelim_results['rankings']
                        ]
                        st.success(f"Preliminary rankings saved! Top seed: {st.session_state.prelim_rankings[0]}")
        
        with col2:
            show_visualization = st.checkbox("Show Bracket Visualization", value=True, key="show_viz")
            show_match_details = st.checkbox("Show Match Details", value=False, key="show_details")
            
            # Advanced options
            with st.expander("Advanced Options"):
                tiebreak_rounds = st.slider("Tie-breaker rounds", 1, 10, 3, 
                                        help="Number of extra rounds to break ties")
                include_third_place = st.checkbox("Include 3rd place match", value=True,
                                                help="Play a match for 3rd place in single elimination")
        
        # Run bracket tournament button
        if st.button("🎯 Run Bracket Tournament", type="primary", width='stretch', key="run_bracket"):
            with st.spinner("Running bracket tournament..."):
                # Prepare strategies based on seeding
                if seeding_method == "Ranked (based on round-robin performance)" and st.session_state.prelim_rankings:
                    # Use preliminary rankings
                    ranked_strategies = []
                    for name in st.session_state.prelim_rankings:
                        strategy = next((s for s in selected_strategies if s.name == name), None)
                        if strategy:
                            ranked_strategies.append(strategy)
                    # Add any strategies not in prelim rankings
                    for strategy in selected_strategies:
                        if strategy not in ranked_strategies:
                            ranked_strategies.append(strategy)
                    tournament_strategies = ranked_strategies
                else:
                    # Random seeding
                    tournament_strategies = selected_strategies.copy()
                    import random
                    random.shuffle(tournament_strategies)
                
                # Run bracket tournament
                bracket = BracketTournament(
                    tournament_strategies,
                    payoff,
                    rounds_per_match=rounds_per_match,
                    noise=noise,
                    bracket_type="single" if bracket_type == "Single Elimination" else "double"
                )
                
                results = bracket.run_tournament()
                bracket_data = bracket.get_bracket_visualization_data(results)
                summary = BracketVisualizer.create_bracket_summary_table(results)
                
                # Store results for radar analysis
                st.session_state.bracket_results = results
                
                # Display championship results
                st.subheader("🏆 Championship Results")
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Champion", summary.get('champion', 'N/A'), 
                            delta="🏆", delta_color="off")
                
                with cols[1]:
                    st.metric("Runner-up", summary.get('runner_up', 'N/A'), 
                            delta="🥈", delta_color="off")
                
                with cols[2]:
                    if summary.get('third_place'):
                        st.metric("3rd Place", summary['third_place'], 
                                delta="🥉", delta_color="off")
                    else:
                        st.metric("3rd Place", "N/A")
                
                with cols[3]:
                    st.metric("Total Matches", summary.get('total_matches', 0))
                
                # Show bracket visualization
                if show_visualization:
                    st.subheader("📊 Bracket Visualization")
                    
                    if bracket_type == "Single Elimination":
                        fig_height = max(8, len(tournament_strategies) * 0.8)
                        fig, ax = plt.subplots(figsize=(14, fig_height))
                        BracketVisualizer.plot_single_elimination_bracket(bracket_data, ax)
                    else:
                        fig, ax = plt.subplots(figsize=(16, 10))
                        BracketVisualizer.plot_double_elimination_bracket(bracket_data, ax)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Download option for bracket image
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save Bracket Image", key="save_bracket"):
                            import io
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                            st.download_button(
                                label="Download Bracket",
                                data=buf.getvalue(),
                                file_name=f"bracket_tournament_{bracket_type.replace(' ', '_')}.png",
                                mime="image/png"
                            )
                
                # Strategy statistics
                st.subheader("📈 Strategy Statistics")
                
                # Convert stats to DataFrame
                stats_data = []
                for name, stats in summary.get('strategy_stats', {}).items():
                    stats_data.append({
                        'Strategy': name,
                        'Matches': stats.get('matches_played', 0),
                        'Wins': stats.get('wins', 0),
                        'Losses': stats.get('losses', 0),
                        'Win Rate': f"{stats.get('win_rate', 0):.1%}",
                        'Avg Score': f"{stats.get('avg_score', 0):.2f}",
                        'Total Score': f"{stats.get('total_score', 0):.1f}"
                    })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Display with sorting options
                    sort_by = st.selectbox("Sort by:", 
                                        ['Wins (desc)', 'Win Rate (desc)', 'Avg Score (desc)', 'Strategy (A-Z)'],
                                        key="sort_stats")
                    
                    if sort_by == 'Wins (desc)':
                        stats_df = stats_df.sort_values('Wins', ascending=False)
                    elif sort_by == 'Win Rate (desc)':
                        stats_df['Win Rate Num'] = stats_df['Win Rate'].str.rstrip('%').astype('float') / 100
                        stats_df = stats_df.sort_values('Win Rate Num', ascending=False)
                        stats_df = stats_df.drop('Win Rate Num', axis=1)
                    elif sort_by == 'Avg Score (desc)':
                        stats_df['Avg Score Num'] = stats_df['Avg Score'].str.replace(',', '').astype('float')
                        stats_df = stats_df.sort_values('Avg Score Num', ascending=False)
                        stats_df = stats_df.drop('Avg Score Num', axis=1)
                    else:
                        stats_df = stats_df.sort_values('Strategy')
                    
                    st.dataframe(stats_df, width='stretch', hide_index=True)
                    
                    # Performance visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Win Distribution")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        strategies = stats_df['Strategy'].tolist()
                        wins = stats_df['Wins'].tolist()
                        
                        colors = ['#FFD700' if name == summary.get('champion') else 
                                '#C0C0C0' if name == summary.get('runner_up') else 
                                '#CD7F32' if name == summary.get('third_place') else 
                                '#1E88E5' for name in strategies]
                        
                        bars = ax.barh(strategies, wins, color=colors)
                        ax.set_xlabel('Number of Wins')
                        ax.set_title('Bracket Tournament Wins')
                        ax.grid(True, alpha=0.3, axis='x')
                        
                        # Add value labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2,
                                f' {width}', va='center')
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        st.subheader("Win Rate vs Avg Score")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Extract numeric values
                        win_rates = [float(w.rstrip('%')) for w in stats_df['Win Rate']]
                        avg_scores = [float(s) for s in stats_df['Avg Score']]
                        
                        scatter = ax.scatter(win_rates, avg_scores, s=100, alpha=0.6, 
                                        c=range(len(strategies)), cmap='viridis')
                        
                        # Highlight top performers
                        if summary.get('champion'):
                            idx = strategies.index(summary['champion'])
                            ax.scatter(win_rates[idx], avg_scores[idx], s=200, 
                                    marker='*', color='gold', label='Champion', edgecolors='black')
                        
                        if summary.get('runner_up') and summary['runner_up'] in strategies:
                            idx = strategies.index(summary['runner_up'])
                            ax.scatter(win_rates[idx], avg_scores[idx], s=150, 
                                    marker='^', color='silver', label='Runner-up', edgecolors='black')
                        
                        ax.set_xlabel('Win Rate (%)')
                        ax.set_ylabel('Average Score')
                        ax.set_title('Performance Scatter Plot')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Add strategy labels (limit to avoid clutter)
                        for i, strategy in enumerate(strategies):
                            ax.annotate(strategy, (win_rates[i], avg_scores[i]), 
                                    fontsize=8, alpha=0.7)
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Show match details if requested
                    if show_match_details and results.get('match_history'):
                        st.subheader("📋 Match Details")
                        
                        # Filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            show_all = st.checkbox("Show all matches", value=True, key="show_all")
                        with col2:
                            if not show_all:
                                match_filter = st.selectbox("Filter by strategy:", 
                                                        ["All"] + sorted(list(summary.get('strategy_stats', {}).keys())),
                                                        key="match_filter")
                        
                        matches_to_show = results['match_history']
                        if not show_all and match_filter != "All":
                            matches_to_show = [m for m in matches_to_show 
                                            if match_filter in [m['strategy1'], m['strategy2']]]
                        
                        for i, match in enumerate(matches_to_show):
                            with st.expander(f"Match {i+1}: {match['strategy1']} vs {match['strategy2']} "
                                        f"(Winner: {match['winner']})"):
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"**{match['strategy1']}**")
                                    st.metric("Score", f"{match['score1']:.2f}")
                                    if match.get('history'):
                                        coop1 = sum(1 for h in match['history'] if h[0] == 'C') / len(match['history'])
                                        st.metric("Cooperation", f"{coop1:.1%}")
                                
                                with col2:
                                    st.markdown(f"**{match['strategy2']}**")
                                    st.metric("Score", f"{match['score2']:.2f}")
                                    if match.get('history'):
                                        coop2 = sum(1 for h in match['history'] if h[1] == 'C') / len(match['history'])
                                        st.metric("Cooperation", f"{coop2:.1%}")
                                
                                with col3:
                                    st.markdown("**Match Info**")
                                    st.write(f"Winner: **{match['winner']}**")
                                    st.write(f"Margin: **{abs(match['score1'] - match['score2']):.2f}**")
                                    if match.get('bracket'):
                                        st.write(f"Bracket: **{match['bracket']}**")
                                
                                # Show round-by-round details
                                if match.get('history') and st.checkbox(f"Show round details for Match {i+1}", 
                                                                    key=f"show_rounds_{i}"):
                                    rounds_data = []
                                    for round_num, (m1, m2, p1, p2) in enumerate(match['history'], 1):
                                        rounds_data.append({
                                            'Round': round_num,
                                            match['strategy1']: m1,
                                            match['strategy2']: m2,
                                            f"{match['strategy1']} Payoff": p1,
                                            f"{match['strategy2']} Payoff": p2
                                        })
                                    
                                    rounds_df = pd.DataFrame(rounds_data)
                                    st.dataframe(rounds_df, width='stretch', hide_index=True)
                                    
                                    # Quick summary
                                    total_rounds = len(match['history'])
                                    coop_rounds = sum(1 for h in match['history'] if h[0] == 'C' and h[1] == 'C')
                                    st.write(f"Mutual Cooperation: {coop_rounds}/{total_rounds} rounds "
                                        f"({coop_rounds/total_rounds:.1%})")
                    
                    # Tournament summary
                    st.subheader("📊 Tournament Summary")
                    
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Total Strategies", len(tournament_strategies))
                        st.metric("Bracket Type", bracket_type)
                    
                    with summary_cols[1]:
                        total_rounds_played = sum(len(m.get('history', [])) for m in results.get('match_history', []))
                        st.metric("Total Rounds Played", total_rounds_played)
                        st.metric("Average Match Length", f"{rounds_per_match} rounds")
                    
                    with summary_cols[2]:
                        if summary.get('strategy_stats'):
                            best_win_rate = max(s.get('win_rate', 0) for s in summary['strategy_stats'].values())
                            best_strategy = [name for name, stats in summary['strategy_stats'].items() 
                                        if stats.get('win_rate', 0) == best_win_rate][0]
                            st.metric("Best Win Rate", f"{best_win_rate:.1%}", best_strategy)
                        
                        highest_avg = max(float(s.get('avg_score', 0)) for s in summary['strategy_stats'].values())
                        highest_strategy = [name for name, stats in summary['strategy_stats'].items() 
                                        if float(stats.get('avg_score', 0)) == highest_avg][0]
                        st.metric("Highest Avg Score", f"{highest_avg:.2f}", highest_strategy)

    with tab6:
        st.markdown('<h2 class="sub-header">Strategy Radar</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666;">Compare strategy performance across multiple dimensions without fixing other strategies. Runs independent simulations for unbiased comparison.</p>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Radar Settings")
            
            # Radar-specific settings - independent from other tabs
            radar_rounds = st.slider("Rounds for Analysis", 10, 100, 30, 
                                    key="radar_rounds",
                                    help="Number of rounds per match for radar analysis")
            radar_noise = st.slider("Noise for Analysis", 0.0, 0.2, 0.05, 
                                   step=0.01, key="radar_noise",
                                   help="Noise level for radar analysis")
            
            # Choose data source
            data_source = st.radio(
                "Data Source",
                ["Run Fresh Simulation", "Use Tournament Results", "Use Bracket Results"],
                index=0,
                key="radar_source",
                help="Select data source for radar visualization"
            )
            
            # Run radar analysis button
            if st.button("🎯 Generate Strategy Radar", type="primary", width='stretch', key="run_radar"):
                with st.spinner("Running comprehensive strategy analysis..."):
                    
                    if data_source == "Use Tournament Results" and st.session_state.tournament_results:
                        # Use existing tournament results
                        radar_results = st.session_state.tournament_results
                        radar_tournament = st.session_state.get('tournament', None)
                    elif data_source == "Use Bracket Results" and st.session_state.bracket_results:
                        # Use bracket results (convert to metrics)
                        radar_results = st.session_state.bracket_results
                        radar_tournament = None
                    else:
                        # Run fresh simulation
                        radar_tournament = Tournament(selected_strategies, payoff,
                                                    rounds_per_match=radar_rounds,
                                                    noise=radar_noise)
                        radar_results = radar_tournament.run_round_robin()
                    
                    # Compute strategy metrics
                    if radar_tournament:
                        metrics_raw = compute_strategy_metrics(radar_results, radar_tournament)
                    else:
                        # Create metrics from bracket results
                        metrics_raw = compute_metrics_from_bracket(radar_results, payoff, rounds_per_match, noise)
                    
                    metrics_normalized = normalize_metrics(metrics_raw)
                    
                    # Store in session state
                    st.session_state.radar_metrics = metrics_normalized
                    st.success(f"✅ Radar analysis complete! {len(metrics_normalized)} strategies analyzed.")
            
            # Display available strategies
            if st.session_state.radar_metrics:
                st.subheader("Strategy Selection")
                st.info(f"✓ {len(st.session_state.radar_metrics)} strategies ready for analysis")
                
                strategy_names = list(st.session_state.radar_metrics.keys())
                strategy_names.sort()
                
                # Single strategy selection
                selected_strategy = st.selectbox(
                    "Single strategy view",
                    strategy_names,
                    key="radar_strategy_select"
                )
                
                # Display strategy description
                strategy_obj = get_strategy_by_name(selected_strategy)
                if strategy_obj:
                    st.caption(f"*{strategy_obj.description}*")
                
                st.markdown("---")
                
                # Compare multiple strategies option
                compare_mode = st.checkbox("📊 Compare multiple strategies", 
                                         key="compare_radar",
                                         value=False,
                                         help="Select multiple strategies to compare on the same radar chart")
                
                selected_strategies_compare = []
                if compare_mode:
                    st.subheader("Compare Strategies")
                    selected_strategies_compare = st.multiselect(
                        "Select 2-4 strategies to compare",
                        strategy_names,
                        default=[strategy_names[0], strategy_names[1]] if len(strategy_names) > 1 else strategy_names,
                        max_selections=4,
                        key="radar_compare_select"
                    )
                    
                    if len(selected_strategies_compare) < 2:
                        st.warning("⚠️ Please select at least 2 strategies to compare")
        
        with col2:
            if st.session_state.radar_metrics:
                st.subheader("Strategy Performance Radar Chart")
                
                with st.container():
                    # Create radar chart based on mode
                    if compare_mode and len(selected_strategies_compare) >= 2:
                        # Generate comparison radar chart
                        fig = plot_strategy_radar_comparison(
                            st.session_state.radar_metrics, 
                            selected_strategies_compare
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Add comparison legend
                            st.markdown("""
                            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem;">
                                <strong>📊 Comparison Legend:</strong><br>
                                Each line represents a different strategy. Use the selection panel to add/remove strategies.
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Single strategy radar
                        selected = st.session_state.get('radar_strategy_select', 
                                                      list(st.session_state.radar_metrics.keys())[0])
                        fig = plot_strategy_radar(st.session_state.radar_metrics, selected)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    # Download radar chart
                    if fig:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Radar Chart", key="save_radar"):
                                import io
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=buf.getvalue(),
                                    file_name=f"strategy_radar_{'comparison' if compare_mode else selected}.png",
                                    mime="image/png"
                                )
                
                # Detailed metrics table
                st.subheader("Detailed Strategy Metrics")
                metrics_df = pd.DataFrame(st.session_state.radar_metrics).T
                
                # Format metrics for display
                formatted_metrics = metrics_df.copy()
                for col in formatted_metrics.columns:
                    if col in ['cooperation_rate', 'forgivingness', 'retaliation']:
                        formatted_metrics[col] = (formatted_metrics[col] * 100).map('{:.1f}%'.format)
                    else:
                        formatted_metrics[col] = formatted_metrics[col].map('{:.3f}'.format)
                
                # Highlight selected strategies
                def highlight_selected(x):
                    styles = [''] * len(x)
                    if compare_mode and len(selected_strategies_compare) > 0:
                        if x.name in selected_strategies_compare:
                            styles = ['background-color: #e3f2fd'] * len(x)
                    elif 'selected_strategy' in locals():
                        if x.name == selected_strategy:
                            styles = ['background-color: #e3f2fd'] * len(x)
                    return styles
                
                st.dataframe(
                    formatted_metrics.style.apply(highlight_selected, axis=1),
                    width='stretch'
                )
                
                # Metric explanations
                with st.expander("📊 Understanding Radar Dimensions"):
                    st.markdown("""
                    **The Strategy Radar visualizes 5 key dimensions:**
                    
                    1. **🎯 Score** - Total points earned in tournament (normalized)
                       * Higher score = more successful strategy
                    
                    2. **🤝 Cooperation Rate** - Frequency of cooperative moves
                       * Higher = more willing to cooperate
                    
                    3. **💪 Robustness** - Performance consistency across different opponents
                       * Higher = more stable performance
                    
                    4. **🛡️ Forgivingness** - Willingness to cooperate after opponent defects
                       * Higher = more forgiving of defection
                    
                    5. **⚔️ Retaliation** - Likelihood to punish defection
                       * Higher = stronger response to defection
                    
                    **All dimensions are normalized (0-1 scale)** for fair comparison.
                    """)
            else:
                st.info("👈 Click 'Generate Strategy Radar' to analyze strategy performance across multiple dimensions.")
                
                # Show preview of what radar does
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="margin-top: 0;">🎯 What is Strategy Radar?</h4>
                    <p>The Strategy Radar provides a comprehensive view of each strategy's strengths and weaknesses:</p>
                    <ul>
                        <li><strong>Independent Analysis:</strong> Runs fresh simulations without interference from other tabs</li>
                        <li><strong>Multi-dimensional:</strong> Evaluates strategies across 5 key metrics</li>
                        <li><strong>Normalized Comparison:</strong> All metrics scaled to 0-1 for fair comparison</li>
                        <li><strong>Strategy Profiles:</strong> Quickly identify if a strategy is cooperative, retaliatory, or robust</li>
                        <li><strong>Compare Mode:</strong> Plot up to 4 strategies on the same radar chart</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    #added by thu for decision tree tab
    with tab7:
        st.markdown('<h2 class="sub-header">Decision Tree Strategy Classification</h2>', 
                unsafe_allow_html=True)

        if st.session_state.radar_metrics:

           if st.button("🌳 Train Decision Tree", type="primary"):
            # Right before preparing data for decision tree
            #st.write("Debug: session data for decision tree")
            #st.write(st.session_state.radar_metrics)   # Or wherever your DF comes from
            #df = prepare_decision_tree_data(st.session_state.radar_metrics)
            #st.write("Columns in DF:", df.columns)


            df = prepare_decision_tree_data(
                st.session_state.radar_metrics
            )

            #clf, features = train_decision_tree(df)
            clf, features, X_train, X_test, y_train, y_test = train_decision_tree(df)
            #Accuracy
            #Decision rules
            st.subheader("Decision Rules")
            rules=export_text(clf, feature_names=features)
            st.text(rules)
            #decision path
            st.subheader("Decision Path Example")
            sample=X_test.iloc[0]
            prediction=clf.predict(sample.to_frame().T)[0]
            st.write("Sample Metrics:")
            st.write(sample)
            st.write("Predicted Label:",prediction)
            #strategy recommendation
            st.subheader("Strategy Recommendation")
            if prediction=="High":
                st.info("Your strategy is highly aggresive. Consider improving stability or forgiveness to avoid risky losses.")
            elif prediction=="Low":
                st.info("Good balance or defensive strategy detected. Small adjustments may optimize results.")
            
            else:
                st.info("Analyze your radar metrics for possible improvments.")
            #df = prepare_decision_tree_data(st.session_state.radar_metrics)
            #st.write(df.head())

            #feature importance
            importance=pd.DataFrame({
                "Feature":features,
                "Importance":clf.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.subheader("Feature Importance")
            st.bar_chart(importance.set_index("Feature"))
            st.subheader("Training Data")
            st.dataframe(df)

            
            

            st.subheader("Decision Tree Structure")

            fig = plot_decision_tree(clf, features)
            st.pyplot(fig)
            plt.close(fig)

            st.success("✅ Decision Tree trained successfully!")

        else:
          st.info("Please generate Strategy Radar first to create metrics data.")
    #added by thu for decision tree ends here
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">'
        '🎮 Strategy Tournament Game | Built with Streamlit and Game Theory | '
        '<a href="https://github.com/yourusername/strategy-tournament">GitHub</a>'
        '</p>',
        unsafe_allow_html=True
    )
    

def plot_strategy_radar_comparison(metrics, strategies_to_compare):
    """Plot multiple strategies on the same radar chart for comparison"""
    if not metrics or len(strategies_to_compare) < 2:
        return None
    
    # Define categories
    categories = ['Score', 'Cooperation\nRate', 'Robustness', 'Forgivingness', 'Retaliation']
    num_vars = len(categories)
    
    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Colors for different strategies
    colors = ['#1E88E5', '#FFC107', '#DC143C', '#2E7D32', '#9C27B0', '#FF8C00', '#00ACC1']
    
    # Plot each strategy
    for i, strategy_name in enumerate(strategies_to_compare):
        if strategy_name in metrics:
            strategy_metrics = metrics[strategy_name]
            
            # Extract values in correct order
            values = [
                strategy_metrics.get('score', 0.5),
                strategy_metrics.get('cooperation_rate', 0.5),
                strategy_metrics.get('robustness', 0.5),
                strategy_metrics.get('forgivingness', 0.5),
                strategy_metrics.get('retaliation', 0.5)
            ]
            values += values[:1]  # Close the loop
            
            # Plot with different color and style
            color = colors[i % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)
    
    # Add title and legend
    ax.set_title('Strategy Performance Comparison', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return fig

def compute_metrics_from_bracket(bracket_results, payoff, rounds_per_match, noise):
    """Helper function to compute strategy metrics from bracket results"""
    metrics = {}
    
    if not bracket_results or 'match_history' not in bracket_results:
        return metrics
    
    # Group matches by strategy
    strategy_matches = {}
    for match in bracket_results['match_history']:
        s1, s2 = match['strategy1'], match['strategy2']
        
        if s1 not in strategy_matches:
            strategy_matches[s1] = []
        if s2 not in strategy_matches:
            strategy_matches[s2] = []
        
        strategy_matches[s1].append({
            'opponent': s2,
            'score': match['score1'],
            'history': match.get('history', [])
        })
        strategy_matches[s2].append({
            'opponent': s1,
            'score': match['score2'],
            'history': match.get('history', [])
        })
    
    # Compute metrics for each strategy
    for strategy_name, matches in strategy_matches.items():
        if not matches:
            continue
        
        # Total score
        total_score = sum(m['score'] for m in matches)
        avg_score = total_score / len(matches)
        
        # Cooperation rate
        all_moves = []
        for match in matches:
            if match['history']:
                # Find which player is the current strategy
                for h in match['history']:
                    all_moves.append(h[0] if match['opponent'] != strategy_name else h[1])
        
        cooperation_rate = sum(1 for m in all_moves if m == 'C') / len(all_moves) if all_moves else 0.5
        
        # Robustness (inverse of score variance)
        scores = [m['score'] for m in matches]
        robustness = 1 / (1 + np.std(scores)) if len(scores) > 1 else 1.0
        
        # Forgivingness and retaliation
        forgivingness = 0.5  # Default
        retaliation = 0.5    # Default
        
        if all_moves and len(matches) > 0:
            # Analyze response patterns
            for match in matches:
                if match['history'] and len(match['history']) > 1:
                    for i in range(1, len(match['history'])):
                        prev_move = match['history'][i-1]
                        curr_move = match['history'][i]
                        
                        # Check if player is the current strategy
                        is_player1 = (match['opponent'] != strategy_name)
                        
                        if is_player1:
                            opponent_prev = prev_move[1]
                            player_curr = curr_move[0]
                        else:
                            opponent_prev = prev_move[0]
                            player_curr = curr_move[1]
                        
                        # Retaliation: defecting after opponent defection
                        if opponent_prev == 'D' and player_curr == 'D':
                            retaliation += 0.1
                        elif opponent_prev == 'D' and player_curr == 'C':
                            retaliation -= 0.1
                        
                        # Forgivingness: cooperating after opponent defection
                        if opponent_prev == 'D' and player_curr == 'C':
                            forgivingness += 0.1
                        elif opponent_prev == 'D' and player_curr == 'D':
                            forgivingness -= 0.1
        
        # Normalize retaliation and forgivingness
        retaliation = max(0, min(1, retaliation))
        forgivingness = max(0, min(1, forgivingness))
        
        metrics[strategy_name] = {
            'score': avg_score / 10,  # Normalize roughly
            'cooperation_rate': cooperation_rate,
            'robustness': robustness,
            'forgivingness': forgivingness,
            'retaliation': retaliation
        }
    
    return metrics

if __name__ == "__main__":
    main()