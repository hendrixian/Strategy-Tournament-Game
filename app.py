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
from analysis import TournamentAnalyzer, analyze_nash_equilibrium

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
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
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
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Tournament", "📈 Evolution", "🎮 Single Match", "📚 Theory"])
    
    # Initialize payoff matrix
    try:
        payoff = PayoffMatrix(game_type, **payoff_kwargs)
    except ValueError as e:
        st.error(f"Invalid payoff parameters: {e}")
        st.stop()
    
    with tab1:
        st.markdown('<h2 class="sub-header">Tournament Results</h2>', unsafe_allow_html=True)
        
        if st.button("🏆 Run Tournament", type="primary", use_container_width=True):
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
                st.dataframe(summary, use_container_width=True, hide_index=True)
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Strategy Performance")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    analyzer.plot_score_distribution(ax)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Cooperation Network")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    analyzer.plot_cooperation_heatmap(results['match_results'], ax)
                    st.pyplot(fig)
                
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
                        history = match_data['history']
                        if history:
                            coop1 = sum(1 for h in history if h[0] == 'C') / len(history)
                            coop2 = sum(1 for h in history if h[1] == 'C') / len(history)
                            st.metric(f"{match_data['strategy1']} Cooperation", 
                                     f"{coop1:.1%}")
                            st.metric(f"{match_data['strategy2']} Cooperation", 
                                     f"{coop2:.1%}")
                    
                    with col2:
                        # Plot match history
                        fig, ax = plt.subplots(figsize=(10, 4))
                        analyzer.plot_match_history(history, ax)
                        st.pyplot(fig)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Evolutionary Dynamics</h2>', unsafe_allow_html=True)
        
        if st.button("🌱 Run Evolutionary Simulation", type="primary", use_container_width=True):
            with st.spinner("Running evolutionary simulation..."):
                # Initialize evolutionary dynamics
                evo = EvolutionaryDynamics(selected_strategies, payoff)
                
                # Run replicator dynamics
                initial_pop = np.ones(len(selected_strategies))
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
                st.dataframe(ess_df, use_container_width=True, hide_index=True)
                
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
                
                with col2:
                    st.subheader("Fitness Evolution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.plot(fitness_history, 'k-', linewidth=2)
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Average Population Fitness')
                    ax.set_title('Evolution of Average Fitness')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
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
            st.dataframe(moves_df, use_container_width=True, hide_index=True)
            
            # Plot match history
            fig, ax = plt.subplots(figsize=(12, 4))
            analyzer = TournamentAnalyzer({})
            analyzer.plot_match_history(history, ax)
            st.pyplot(fig)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Game Theory Concepts</h2>', unsafe_allow_html=True)
        
        # Game analysis
        st.subheader("Game Analysis")
        nash_table = analyze_nash_equilibrium(payoff)
        st.dataframe(nash_table, use_container_width=True, hide_index=True)
        
        # Payoff matrix visualization
        st.subheader("Payoff Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        analyzer = TournamentAnalyzer({})
        analyzer.plot_payoff_matrix(payoff, ax)
        st.pyplot(fig)
        
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

if __name__ == "__main__":
    main()