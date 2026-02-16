"""
Analysis and visualization of tournament results.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
import pandas as pd

class TournamentAnalyzer:
    """Analyze and visualize tournament results"""
    
    def __init__(self, tournament_results: Dict[str, Any]):
        self.results = tournament_results
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of tournament results"""
        scores = self.results.get('scores', {})
        rankings = self.results.get('rankings', [])
        
        # Get cooperation rates if available
        cooperation_rates = self.results.get('cooperation_rates', {})
        
        data = []
        for rank, (name, score) in enumerate(rankings, 1):
            row = {
                'Rank': rank,
                'Strategy': name,
                'Average Score': f"{score:.2f}",
                'Cooperation Rate': f"{cooperation_rates.get(name, 0):.2%}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def plot_score_distribution(self, ax=None):
        """Plot distribution of scores"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = self.results.get('scores', {})
        if not scores:
            return ax
        
        names = list(scores.keys())
        values = [scores[name] for name in names]
        
        # Sort by score
        sorted_indices = np.argsort(values)
        names = [names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        bars = ax.barh(names, values, color='steelblue')
        ax.set_xlabel('Average Score')
        ax.set_title('Strategy Performance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {width:.2f}', va='center')
        
        return ax
    
    def plot_cooperation_network(self, match_results: List[Dict[str, Any]], ax=None):
        """Plot a directed cooperation network between strategies."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract unique strategies
        strategies = set()
        for match in match_results:
            strategies.add(match['strategy1'])
            strategies.add(match['strategy2'])
        
        strategies = sorted(strategies)
        n = len(strategies)
        
        if n == 0:
            ax.set_title('Cooperation Network')
            ax.axis('off')
            return ax
        
        # Map strategy names to indices
        strategy_index = {name: i for i, name in enumerate(strategies)}
        

        # Build directed edge list from match histories
        edges = []
        for match in match_results:
            s1 = match['strategy1']
            s2 = match['strategy2']
            history = match['history']
            if not history:
                continue
            
            i = strategy_index[s1]
            j = strategy_index[s2]
            
            coop_rate_1_to_2 = sum(1 for h in history if h[0] == 'C') / len(history)
            coop_rate_2_to_1 = sum(1 for h in history if h[1] == 'C') / len(history)

            edges.append((i, j, coop_rate_1_to_2))
            edges.append((j, i, coop_rate_2_to_1))

        # Place nodes on a circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0
        positions = {
            idx: (radius * np.cos(angle), radius * np.sin(angle))
            for idx, angle in enumerate(angles)
        }

        # Draw directed edges (color and thickness encode cooperation rate)
        cmap = plt.cm.RdYlGn
        for src, dst, weight in edges:
            if src == dst:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            ax.annotate(
                '',
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->',
                    color=cmap(weight),
                    lw=1.0 + 3.0 * weight,
                    alpha=0.75,
                    shrinkA=18,
                    shrinkB=18,
                    connectionstyle='arc3,rad=0.12'
                )
            )

        # Draw nodes
        node_x = [positions[i][0] for i in range(n)]
        node_y = [positions[i][1] for i in range(n)]
        ax.scatter(node_x, node_y, s=1200, c='lightsteelblue', edgecolors='black', zorder=3)

        
        for i, name in enumerate(strategies):
            x, y = positions[i]
            ax.text(x, y, name, ha='center', va='center', fontsize=8, zorder=4)

        # Colorbar legend for edge colors
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Cooperation Rate (edge direction)')

        ax.set_title('Directed Cooperation Network')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        return ax
    
    def plot_evolutionary_dynamics(self, pop_history: np.ndarray, 
                                 strategy_names: List[str], ax=None):
        """Plot evolutionary dynamics over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = pop_history.shape[0]
        
        # Plot each strategy's frequency
        for i, name in enumerate(strategy_names):
            ax.plot(range(generations), pop_history[:, i], 
                   label=name, linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Frequency')
        ax.set_title('Evolutionary Dynamics')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_payoff_matrix(self, payoff_matrix, ax=None):
        """Visualize the payoff matrix"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create table data
        moves = ['Cooperate', 'Defect']
        data = []
        
        for i, m1 in enumerate(moves):
            for j, m2 in enumerate(moves):
                payoff = payoff_matrix.get_payoff('C' if i == 0 else 'D', 
                                                'C' if j == 0 else 'D')
                data.append([
                    m1, m2, 
                    f"{payoff[0]:.1f}", 
                    f"{payoff[1]:.1f}"
                ])
        
        # Create table
        table = ax.table(cellText=data,
                        colLabels=['Player 1', 'Player 2', 'P1 Payoff', 'P2 Payoff'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.axis('off')
        ax.set_title('Payoff Matrix', fontsize=14, pad=20)
        
        return ax
    
    def plot_match_history(self, history: List[Tuple], ax=None):
        """Plot the history of moves in a match"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        if not history:
            return ax
        
        rounds = len(history)
        
        # Extract moves and payoffs
        player1_moves = [h[0] for h in history]
        player2_moves = [h[1] for h in history]
        player1_payoffs = [h[2] for h in history]
        player2_payoffs = [h[3] for h in history]
        
        # Convert moves to numeric for plotting
        move_to_num = {'C': 1, 'D': 0}
        p1_numeric = [move_to_num[m] for m in player1_moves]
        p2_numeric = [move_to_num[m] for m in player2_moves]
        
        # Plot moves
        ax.plot(range(rounds), p1_numeric, 'o-', label='Player 1', linewidth=2, markersize=8)
        ax.plot(range(rounds), p2_numeric, 's-', label='Player 2', linewidth=2, markersize=8)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Defect (D)', 'Cooperate (C)'])
        ax.set_xlabel('Round')
        ax.set_ylabel('Move')
        ax.set_title('Match History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add payoff information as text
        total_p1 = sum(player1_payoffs)
        total_p2 = sum(player2_payoffs)
        ax.text(0.02, 0.98, f'Total Payoffs: P1={total_p1}, P2={total_p2}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax

def analyze_nash_equilibrium(payoff_matrix) -> pd.DataFrame:
    """Create a table analyzing Nash equilibria"""
    analysis = payoff_matrix.analyze_nash_equilibrium()
    
    data = []
    
    # Game type
    data.append(['Game Type', analysis['game_type']])
    
    # Payoffs - format as strings
    payoffs = analysis['payoffs']
    if 'T' in payoffs:
        data.append(['T (Temptation)', f"{payoffs['T']:.2f}"])
        data.append(['R (Reward)', f"{payoffs['R']:.2f}"])
        data.append(['P (Punishment)', f"{payoffs['P']:.2f}"])
        data.append(['S (Sucker)', f"{payoffs['S']:.2f}"])
    else:
        data.append(['Win Payoff', f"{payoffs['win']:.2f}"])
        data.append(['Lose Payoff', f"{payoffs['lose']:.2f}"])
    
    # Nash equilibria
    nash_str = ', '.join([f"({e[0]},{e[1]})" for e in analysis['pure_nash_equilibria']])
    data.append(['Pure Nash Equilibria', nash_str if nash_str else 'None'])
    
    # Mixed equilibrium
    if analysis['mixed_equilibrium']:
        mixed = analysis['mixed_equilibrium']
        data.append(['Mixed Strategy p(C)', f"{mixed['p_cooperate']:.3f}"])
        data.append(['Mixed Strategy Payoff', f"{mixed['payoff']:.3f}"])
    else:
        data.append(['Mixed Equilibrium', 'None'])
    
    # Dominant strategies
    dom = analysis['dominant_strategy']
    row_dom = ', '.join(dom['row']) if dom['row'] else 'None'
    col_dom = ', '.join(dom['column']) if dom['column'] else 'None'
    data.append(['Row Player Dominant', row_dom])
    data.append(['Column Player Dominant', col_dom])
    
    return pd.DataFrame(data, columns=['Property', 'Value'])

def compute_strategy_metrics(results, tournament):
    metrics = {}

    scores = results["scores"]                 # avg score per strategy
    cooperation_rates = tournament.get_cooperation_rates()
    detailed_history = results["detailed_history"]

    for name in scores.keys():
        history = detailed_history.get(name, [])

        moves = len(history)
        retaliation = 0
        forgiveness = 0

        # Simple heuristics for radar dimensions
        if moves > 1:
            for i in range(1, len(history)):
                prev_opp_move = history[i-1][1]
                curr_move = history[i][0]

                if prev_opp_move == "D" and curr_move == "D":
                    retaliation += 1
                if prev_opp_move == "D" and curr_move == "C":
                    forgiveness += 1

            retaliation_rate = retaliation / moves
            forgiveness_rate = forgiveness / moves
        else:
            retaliation_rate = 0
            forgiveness_rate = 0

        metrics[name] = {
            "avg_payoff": scores[name],                   # already averaged
            "cooperation_rate": cooperation_rates.get(name, 0),
            "retaliation": retaliation_rate,
            "forgiveness": forgiveness_rate,
            "stability": 0.5,                              # placeholder (from evolution.py later)
            "win_rate": 0.0                                # optional, you can compute later
        }

    return metrics

def plot_strategy_radar(metrics, strategy_name):
    labels = list(metrics[strategy_name].keys())
    values = list(metrics[strategy_name].values())

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 1)  # origin-based scaling
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Strategy Radar: {strategy_name}")
    return fig

def normalize_metrics(metrics):
    normalized = {}
    
    keys = list(next(iter(metrics.values())).keys())

    mins = {k: min(m[k] for m in metrics.values()) for k in keys}
    maxs = {k: max(m[k] for m in metrics.values()) for k in keys}

    for name, m in metrics.items():
        normalized[name] = {}
        for k in keys:
            if maxs[k] - mins[k] == 0:
                normalized[name][k] = 0.5
            else:
                normalized[name][k] = (m[k] - mins[k]) / (maxs[k] - mins[k])
    return normalized
