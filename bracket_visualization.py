"""
Visualization functions for tournament brackets.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any


class BracketVisualizer:
    """Visualize tournament brackets"""
    
    @staticmethod
    def plot_single_elimination_bracket(bracket_data: Dict[str, Any], ax=None, figsize=(15, 10)):
        """Plot single elimination bracket"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        rounds = bracket_data['rounds']
        n_rounds = len(rounds)
        
        # Calculate positions
        max_matches = max(len(round['matches']) for round in rounds)
        y_positions = np.linspace(0, 1, max_matches * 2 + 1)[1::2]
        
        # Draw each round
        for round_idx, round_info in enumerate(rounds):
            round_num = round_info['round_number']
            matches = round_info['matches']
            
            x = round_idx * 0.2
            round_y_positions = y_positions[:len(matches)]
            
            # Draw matches in this round
            for match_idx, (match, y) in enumerate(zip(matches, round_y_positions)):
                # Draw match box
                box = patches.Rectangle((x-0.08, y-0.03), 0.16, 0.06,
                                       linewidth=1, edgecolor='black', facecolor='white')
                ax.add_patch(box)
                
                # Add strategy names
                ax.text(x-0.075, y+0.01, match['strategy1'], fontsize=8,
                       va='center', ha='left', fontweight='bold' if match['winner'] == match['strategy1'] else 'normal')
                ax.text(x-0.075, y-0.01, match['strategy2'], fontsize=8,
                       va='center', ha='left', fontweight='bold' if match['winner'] == match['strategy2'] else 'normal')
                
                # Add scores if played
                if match['played']:
                    ax.text(x+0.07, y+0.01, f"{match['score1']:.1f}", fontsize=8,
                           va='center', ha='right')
                    ax.text(x+0.07, y-0.01, f"{match['score2']:.1f}", fontsize=8,
                           va='center', ha='right')
                
                # Draw connection to next round
                if round_idx < n_rounds - 1:
                    next_x = (round_idx + 1) * 0.2
                    next_match_idx = match_idx // 2
                    if next_match_idx < len(rounds[round_idx + 1]['matches']):
                        next_y = y_positions[next_match_idx * 2]  # Adjusted for next round
                        
                        # Draw vertical line
                        ax.plot([x+0.08, x+0.12], [y, y], 'k-', linewidth=1)
                        
                        # Draw horizontal line
                        ax.plot([x+0.12, next_x-0.08], [y, next_y], 'k-', linewidth=1)
        
        # Add round labels
        for round_idx in range(n_rounds):
            ax.text(round_idx * 0.2, 1.05, f"Round {round_idx + 1}",
                   fontsize=10, fontweight='bold', ha='center')
        
        # Add champion
        if bracket_data['champion']:
            ax.text((n_rounds-1) * 0.2 + 0.1, -0.05,
                   f"🏆 Champion: {bracket_data['champion']}",
                   fontsize=12, fontweight='bold', ha='center', color='gold')
        
        if bracket_data['runner_up']:
            ax.text((n_rounds-1) * 0.2 + 0.1, -0.1,
                   f"🥈 Runner-up: {bracket_data['runner_up']}",
                   fontsize=10, ha='center', color='silver')
        
        if bracket_data.get('third_place'):
            ax.text((n_rounds-1) * 0.2 + 0.1, -0.15,
                   f"🥉 3rd Place: {bracket_data['third_place']}",
                   fontsize=10, ha='center', color='#CD7F32')
        
        ax.set_xlim(-0.1, n_rounds * 0.2 + 0.1)
        ax.set_ylim(-0.2, 1.1)
        ax.axis('off')
        ax.set_title('Single Elimination Tournament Bracket', fontsize=14, fontweight='bold')
        
        return ax
    
    @staticmethod
    def plot_double_elimination_bracket(bracket_data: Dict[str, Any], ax=None, figsize=(20, 12)):
        """Plot double elimination bracket"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        rounds = bracket_data['rounds']
        
        # Separate winners and losers brackets
        winners_x = 0
        losers_x = 0.5
        
        # Track positions
        y_positions = {}
        match_counter = 0
        
        for round_num, round_info in enumerate(rounds):
            # Winners bracket matches
            for match in round_info.get('winners_matches', []):
                y = 1 - (match_counter * 0.1)
                y_positions[match['match_id']] = y
                
                # Draw match
                box = patches.Rectangle((winners_x-0.04, y-0.02), 0.08, 0.04,
                                       linewidth=1, edgecolor='blue', facecolor='lightblue')
                ax.add_patch(box)
                
                # Add text
                ax.text(winners_x, y, f"{match['strategy1']} vs {match['strategy2']}",
                       fontsize=6, va='center', ha='center')
                ax.text(winners_x, y-0.015, f"Winner: {match['winner']}",
                       fontsize=5, va='center', ha='center', color='green')
                
                match_counter += 1
            
            # Losers bracket matches
            for match in round_info.get('losers_matches', []):
                y = 1 - (match_counter * 0.1)
                y_positions[match['match_id']] = y
                
                # Draw match
                box = patches.Rectangle((losers_x-0.04, y-0.02), 0.08, 0.04,
                                       linewidth=1, edgecolor='red', facecolor='lightcoral')
                ax.add_patch(box)
                
                # Add text
                ax.text(losers_x, y, f"{match['strategy1']} vs {match['strategy2']}",
                       fontsize=6, va='center', ha='center')
                ax.text(losers_x, y-0.015, f"Winner: {match['winner']}",
                       fontsize=5, va='center', ha='center', color='green')
                
                match_counter += 1
        
        # Add labels
        ax.text(winners_x, 1.05, "Winners Bracket", fontsize=10,
               fontweight='bold', ha='center', color='blue')
        ax.text(losers_x, 1.05, "Losers Bracket", fontsize=10,
               fontweight='bold', ha='center', color='red')
        
        # Add champion
        if bracket_data['champion']:
            ax.text(0.5, -0.05, f"🏆 Champion: {bracket_data['champion']}",
                   fontsize=12, fontweight='bold', ha='center', color='gold')
        
        if bracket_data['runner_up']:
            ax.text(0.5, -0.1, f"🥈 Runner-up: {bracket_data['runner_up']}",
                   fontsize=10, ha='center', color='silver')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.1)
        ax.axis('off')
        ax.set_title('Double Elimination Tournament Bracket', fontsize=14, fontweight='bold')
        
        return ax
    
    @staticmethod
    def create_bracket_summary_table(results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for bracket tournament"""
        match_history = results.get('match_history', [])
        
        # Calculate strategy statistics
        strategy_stats = {}
        for match in match_history:
            for strategy_name in [match['strategy1'], match['strategy2']]:
                if strategy_name != "BYE":
                    if strategy_name not in strategy_stats:
                        strategy_stats[strategy_name] = {
                            'wins': 0,
                            'losses': 0,
                            'total_score': 0,
                            'matches_played': 0
                        }
                    
                    # Check if this strategy won
                    if match['winner'] == strategy_name:
                        strategy_stats[strategy_name]['wins'] += 1
                        strategy_stats[strategy_name]['total_score'] += match['score1'] if match['strategy1'] == strategy_name else match['score2']
                    else:
                        strategy_stats[strategy_name]['losses'] += 1
                        strategy_stats[strategy_name]['total_score'] += match['score2'] if match['strategy1'] == strategy_name else match['score1']
                    
                    strategy_stats[strategy_name]['matches_played'] += 1
        
        # Calculate averages
        for stats in strategy_stats.values():
            if stats['matches_played'] > 0:
                stats['avg_score'] = stats['total_score'] / stats['matches_played']
                stats['win_rate'] = stats['wins'] / stats['matches_played']
        
        # Sort by wins
        sorted_stats = sorted(strategy_stats.items(),
                            key=lambda x: (x[1]['wins'], x[1]['avg_score']),
                            reverse=True)
        
        return {
            'champion': results.get('champion'),
            'runner_up': results.get('runner_up'),
            'third_place': results.get('third_place'),
            'strategy_stats': dict(sorted_stats),
            'total_matches': len(match_history),
            'total_strategies': len(strategy_stats)
        }