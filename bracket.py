"""
Tournament bracket (single and double elimination) implementation.
"""
import random
import math
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from strategies import Strategy
from payoff import PayoffMatrix

# Define ByeStrategy right here to avoid import issues
class ByeStrategy(Strategy):
    """A placeholder strategy for byes that automatically loses"""
    def __init__(self):
        super().__init__("BYE")
    
    def decide(self, history, opponent):
        return 'C'  # Always cooperate
    
    def reset(self):
        pass  # No state to reset
    
    def __repr__(self):
        return "BYE"
    
    def __str__(self):
        return "BYE"


class BracketTournament:
    """Manage single and double elimination tournaments"""
    
    def __init__(self, strategies: List[Strategy], payoff_matrix: PayoffMatrix,
                 rounds_per_match: int = 10, noise: float = 0.0,
                 bracket_type: str = "single"):
        """
        Initialize bracket tournament.
        
        Args:
            strategies: List of strategy instances
            payoff_matrix: Payoff matrix for the game
            rounds_per_match: Number of rounds per match
            noise: Probability of move being flipped
            bracket_type: "single" for single elimination, "double" for double elimination
        """
        # Validate strategies count for double elimination
        if bracket_type == "double" and len(strategies) % 2 != 0:
            raise ValueError(f"Double elimination requires an even number of strategies. Got {len(strategies)}")
        
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix
        self.rounds_per_match = rounds_per_match
        self.noise = noise
        self.bracket_type = bracket_type
        
        # Tournament state
        self.winners_bracket = []
        self.losers_bracket = []
        self.final_rounds = []
        self.champion = None
        self.runner_up = None
        self.third_place = None
        self.match_history = []
        
    def apply_noise(self, move: str) -> str:
        """Apply noise to a move"""
        if random.random() < self.noise:
            return 'C' if move == 'D' else 'D'
        return move
    
    def play_match(self, strategy1: Strategy, strategy2: Strategy) -> Tuple[float, float, List]:
        """Play a match between two strategies"""
        # Handle byes (None or ByeStrategy) - this prevents the .reset() error
        if strategy1 is None or strategy2 is None or isinstance(strategy1, ByeStrategy) or isinstance(strategy2, ByeStrategy):
            if strategy1 is None or isinstance(strategy1, ByeStrategy):
                # Strategy2 wins by forfeit
                return 0, 1000, []
            else:
                # Strategy1 wins by forfeit
                return 1000, 0, []
        
        # Now it's safe to call .reset() because both strategies are valid
        try:
            strategy1.reset()
        except AttributeError:
            pass  # Some strategies might not have reset method
        
        try:
            strategy2.reset()
        except AttributeError:
            pass  # Some strategies might not have reset method
        
        history = []
        move_history = []
        total_score1 = 0
        total_score2 = 0
        
        for round_num in range(self.rounds_per_match):
            try:
                move1 = strategy1.decide(move_history, strategy2.name)
            except:
                move1 = 'C'  # Default to cooperate if decide fails
            
            try:
                move2 = strategy2.decide([(h[1], h[0]) for h in move_history], strategy1.name)
            except:
                move2 = 'C'  # Default to cooperate if decide fails
            
            move1 = self.apply_noise(move1)
            move2 = self.apply_noise(move2)
            
            payoff1, payoff2 = self.payoff_matrix.get_payoff(move1, move2)
            
            total_score1 += payoff1
            total_score2 += payoff2
            
            history.append((move1, move2, payoff1, payoff2))
            move_history.append((move1, move2))
        
        return total_score1, total_score2, history
    
    def determine_winner(self, score1: float, score2: float, 
                        strategy1: Strategy, strategy2: Strategy) -> Dict[str, Any]:
        """Determine match winner with tie-breaking"""
        # Handle cases where one strategy is a ByeStrategy or None
        if strategy1 is None or isinstance(strategy1, ByeStrategy):
            return {
                'winner': strategy2,
                'loser': strategy1,
                'score1': score1,
                'score2': score2
            }
        if strategy2 is None or isinstance(strategy2, ByeStrategy):
            return {
                'winner': strategy1,
                'loser': strategy2,
                'score1': score1,
                'score2': score2
            }
        
        winner = None
        loser = None
        
        if score1 > score2:
            winner = strategy1
            loser = strategy2
        elif score2 > score1:
            winner = strategy2
            loser = strategy1
        else:
            # Tie-breaking: play extra round(s)
            extra_rounds = 3
            extra_score1 = 0
            extra_score2 = 0
            
            # Get current history for tie-breaker
            move_history = []
            for i in range(extra_rounds):
                try:
                    move1 = strategy1.decide(move_history, strategy2.name)
                except:
                    move1 = 'C'
                
                try:
                    move2 = strategy2.decide([(h[1], h[0]) for h in move_history], strategy1.name)
                except:
                    move2 = 'C'
                
                payoff1, payoff2 = self.payoff_matrix.get_payoff(move1, move2)
                extra_score1 += payoff1
                extra_score2 += payoff2
                move_history.append((move1, move2))
            
            if extra_score1 > extra_score2:
                winner = strategy1
                loser = strategy2
            elif extra_score2 > extra_score1:
                winner = strategy2
                loser = strategy1
            else:
                # Still tied - random choice
                winner = random.choice([strategy1, strategy2])
                loser = strategy2 if winner == strategy1 else strategy1
        
        return {
            'winner': winner,
            'loser': loser,
            'score1': score1,
            'score2': score2
        }
    
    def create_seeding(self, shuffle: bool = True) -> List[Strategy]:
        """Create tournament seeding (optionally shuffled)"""
        seeded_strategies = self.strategies.copy()
        if shuffle:
            random.shuffle(seeded_strategies)
        return seeded_strategies
    
    def create_initial_bracket_single(self, seeded_strategies: List[Strategy]) -> List[Dict[str, Any]]:
        """Create initial bracket matches for SINGLE elimination"""
        n = len(seeded_strategies)
        
        # Handle odd number of strategies - use ByeStrategy
        if n % 2 == 1:
            seeded_strategies.append(ByeStrategy())
            n = len(seeded_strategies)
        
        bracket = []
        for i in range(0, n, 2):
            match = {
                'match_id': len(bracket),
                'strategy1': seeded_strategies[i],
                'strategy2': seeded_strategies[i + 1],
                'winner': None,
                'loser': None,
                'played': False,
                'scores': (0, 0),
                'next_match': None,
                'is_bye': isinstance(seeded_strategies[i], ByeStrategy) or isinstance(seeded_strategies[i + 1], ByeStrategy)
            }
            bracket.append(match)
        
        return bracket
    
    def create_initial_bracket_double(self, seeded_strategies: List[Strategy]) -> List[Dict[str, Any]]:
        """Create initial bracket matches for DOUBLE elimination"""
        n = len(seeded_strategies)
        
        # Double elimination MUST have even number
        if n % 2 == 1:
            raise ValueError(f"Double elimination requires even number of strategies. Got {n}")
        
        bracket = []
        for i in range(0, n, 2):
            match = {
                'match_id': len(bracket),
                'strategy1': seeded_strategies[i],
                'strategy2': seeded_strategies[i + 1],
                'winner': None,
                'loser': None,
                'played': False,
                'scores': (0, 0),
                'next_match': None,
                'is_bye': False  # No byes in double elimination
            }
            bracket.append(match)
        
        return bracket
    
    def run_single_elimination(self) -> Dict[str, Any]:
        """Run single elimination tournament"""
        # Create seeding
        seeded_strategies = self.create_seeding()
        
        # Create initial bracket - USE SINGLE VERSION
        current_round = self.create_initial_bracket_single(seeded_strategies)
        round_number = 1
        all_rounds = [current_round.copy()]
        
        # Track eliminated strategies
        eliminated = []
        self.match_history = []  # Reset match history
        
        while len(current_round) > 1:
            next_round = []
            round_results = []
            
            # Play all matches in current round
            for i, match in enumerate(current_round):
                if match['is_bye']:
                    # Handle bye - advance the non-bye strategy
                    if isinstance(match['strategy1'], ByeStrategy):
                        winner = match['strategy2']
                        loser = match['strategy1']
                        score1, score2 = 0, 1000
                    else:
                        winner = match['strategy1']
                        loser = match['strategy2']
                        score1, score2 = 1000, 0
                    
                    # SAFELY get names with None checking
                    strategy1_name = "BYE"
                    if match['strategy1'] is not None:
                        if hasattr(match['strategy1'], 'name'):
                            strategy1_name = match['strategy1'].name
                        else:
                            strategy1_name = str(match['strategy1'])
                    
                    strategy2_name = "BYE"
                    if match['strategy2'] is not None:
                        if hasattr(match['strategy2'], 'name'):
                            strategy2_name = match['strategy2'].name
                        else:
                            strategy2_name = str(match['strategy2'])
                    
                    winner_name = "BYE"
                    if winner is not None:
                        if hasattr(winner, 'name'):
                            winner_name = winner.name
                        else:
                            winner_name = str(winner)
                    
                    loser_name = "BYE"
                    if loser is not None:
                        if hasattr(loser, 'name'):
                            loser_name = loser.name
                        else:
                            loser_name = str(loser)
                    
                    match_result = {
                        'match_id': match['match_id'],
                        'round': round_number,
                        'strategy1': strategy1_name,
                        'strategy2': strategy2_name,
                        'winner': winner_name,
                        'loser': loser_name,
                        'score1': score1,
                        'score2': score2,
                        'is_bye': True
                    }
                    round_results.append(match_result)
                    self.match_history.append(match_result)
                    
                    # Create next match slot if needed
                    if len(next_round) * 2 <= i:
                        next_round.append({
                            'match_id': len(next_round),
                            'strategy1': None,
                            'strategy2': None,
                            'winner': None,
                            'loser': None,
                            'played': False,
                            'scores': (0, 0),
                            'next_match': None,
                            'is_bye': False
                        })
                    
                    # Place winner in next round
                    if i % 2 == 0:
                        next_round[i // 2]['strategy1'] = winner
                    else:
                        next_round[i // 2]['strategy2'] = winner
                        
                else:
                    # Play actual match
                    s1 = match['strategy1']
                    s2 = match['strategy2']
                    
                    # Skip if either strategy is None (shouldn't happen in non-bye matches)
                    if s1 is None or s2 is None:
                        continue
                    
                    score1, score2, history = self.play_match(s1, s2)
                    
                    # Determine winner
                    result = self.determine_winner(score1, score2, s1, s2)
                    
                    match['winner'] = result['winner']
                    match['loser'] = result['loser']
                    match['scores'] = (score1, score2)
                    match['played'] = True
                    
                    # SAFELY get names with None checking
                    s1_name = "Unknown"
                    if s1 is not None:
                        if hasattr(s1, 'name'):
                            s1_name = s1.name
                        else:
                            s1_name = str(s1)
                    
                    s2_name = "Unknown"
                    if s2 is not None:
                        if hasattr(s2, 'name'):
                            s2_name = s2.name
                        else:
                            s2_name = str(s2)
                    
                    winner_name = "Unknown"
                    if result['winner'] is not None:
                        if hasattr(result['winner'], 'name'):
                            winner_name = result['winner'].name
                        else:
                            winner_name = str(result['winner'])
                    
                    loser_name = "Unknown"
                    if result['loser'] is not None:
                        if hasattr(result['loser'], 'name'):
                            loser_name = result['loser'].name
                        else:
                            loser_name = str(result['loser'])
                    
                    # Record match result
                    match_result = {
                        'match_id': match['match_id'],
                        'round': round_number,
                        'strategy1': s1_name,
                        'strategy2': s2_name,
                        'winner': winner_name,
                        'loser': loser_name,
                        'score1': score1,
                        'score2': score2,
                        'history': history,
                        'is_bye': False
                    }
                    round_results.append(match_result)
                    self.match_history.append(match_result)
                    
                    # Add loser to eliminated list (skip ByeStrategy)
                    if result['loser'] is not None and not isinstance(result['loser'], ByeStrategy):
                        eliminated.append({
                            'strategy': result['loser'],
                            'eliminated_in_round': round_number,
                            'score': result['score2'] if result['loser'] == s2 else result['score1']
                        })
                    
                    # Create next match slot if needed
                    if len(next_round) * 2 <= i:
                        next_round.append({
                            'match_id': len(next_round),
                            'strategy1': None,
                            'strategy2': None,
                            'winner': None,
                            'loser': None,
                            'played': False,
                            'scores': (0, 0),
                            'next_match': None,
                            'is_bye': False
                        })
                    
                    # Place winner in next round
                    if i % 2 == 0:
                        next_round[i // 2]['strategy1'] = result['winner']
                    else:
                        next_round[i // 2]['strategy2'] = result['winner']
            
            # Prepare for next round
            current_round = next_round
            round_number += 1
            
            if current_round:
                all_rounds.append(current_round.copy())
        
        # PLAY THE FINAL MATCH
        if current_round and len(current_round) == 1:
            final_match = current_round[0]
            
            # Make sure both strategies exist for the final
            if final_match['strategy1'] and final_match['strategy2']:
                s1 = final_match['strategy1']
                s2 = final_match['strategy2']
                
                # CRITICAL FIX: If final match is between two ByeStrategies, this is a problem
                if isinstance(s1, ByeStrategy) and isinstance(s2, ByeStrategy):
                    # This should never happen - something went wrong
                    # Find the last real strategy that was eliminated
                    real_strategies = [s for s in self.strategies if not isinstance(s, ByeStrategy)]
                    if real_strategies:
                        # Pick the one with highest score from match history as champion
                        champion = real_strategies[0]  # Default
                        runner_up = real_strategies[1] if len(real_strategies) > 1 else None
                    else:
                        champion = None
                        runner_up = None
                else:
                    # Play the final match normally
                    score1, score2, history = self.play_match(s1, s2)
                    result = self.determine_winner(score1, score2, s1, s2)
                    
                    final_match['winner'] = result['winner']
                    final_match['loser'] = result['loser']
                    final_match['scores'] = (score1, score2)
                    final_match['played'] = True
                    
                    # SAFELY get names
                    s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                    s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                    winner_name = result['winner'].name if hasattr(result['winner'], 'name') else str(result['winner'])
                    loser_name = result['loser'].name if hasattr(result['loser'], 'name') else str(result['loser'])
                    
                    # Record final match
                    match_result = {
                        'match_id': final_match['match_id'],
                        'round': round_number,
                        'strategy1': s1_name,
                        'strategy2': s2_name,
                        'winner': winner_name,
                        'loser': loser_name,
                        'score1': score1,
                        'score2': score2,
                        'history': history,
                        'is_bye': False
                    }
                    self.match_history.append(match_result)
                    
                    champion = result['winner']
                    runner_up = result['loser']
            else:
                # Handle case where final has a bye
                if final_match['strategy1'] and not isinstance(final_match['strategy1'], ByeStrategy):
                    champion = final_match['strategy1']
                    runner_up = None
                elif final_match['strategy2'] and not isinstance(final_match['strategy2'], ByeStrategy):
                    champion = final_match['strategy2']
                    runner_up = None
                else:
                    champion = None
                    runner_up = None
        else:
            champion = None
            runner_up = None
        
        # Set champion and runner-up (filter out ByeStrategy)
        self.champion = None
        self.runner_up = None
        
        # Find the real champion (first non-ByeStrategy)
        if champion is not None and not isinstance(champion, ByeStrategy):
            self.champion = champion
        else:
            # Look for the best performing real strategy from match history
            real_winners = {}
            for match in self.match_history:
                if match.get('winner') and match['winner'] != "BYE" and match['winner'] != "Unknown":
                    winner_name = match['winner']
                    # Find the actual strategy object
                    for s in self.strategies:
                        if s.name == winner_name and not isinstance(s, ByeStrategy):
                            real_winners[s.name] = real_winners.get(s.name, 0) + 1
                            break
            
            if real_winners:
                # Pick the strategy with most wins
                best_winner = max(real_winners.items(), key=lambda x: x[1])[0]
                for s in self.strategies:
                    if s.name == best_winner and not isinstance(s, ByeStrategy):
                        self.champion = s
                        break
        
        # Find runner-up
        if runner_up is not None and not isinstance(runner_up, ByeStrategy):
            self.runner_up = runner_up
        elif self.champion:
            # Find opponent in final match
            for match in self.match_history:
                if match.get('round') == round_number - 1 and match.get('winner') == self.champion.name:
                    # This is the final match, runner-up is the loser
                    loser_name = match.get('loser')
                    if loser_name and loser_name != "BYE":
                        for s in self.strategies:
                            if s.name == loser_name and not isinstance(s, ByeStrategy):
                                self.runner_up = s
                                break
                    break
        
        # Determine 3rd place - find semifinal losers
        semifinal_losers = []
        
        # Look for matches in the second-to-last round where losers were eliminated
        if len(all_rounds) >= 2:
            semifinal_round = all_rounds[-2]  # Second-to-last round (semifinals)
            
            for match in semifinal_round:
                if match.get('loser') and match['loser'] != self.runner_up:
                    # Skip ByeStrategy and None
                    if match['loser'] is not None and not isinstance(match['loser'], ByeStrategy):
                        # This is a semifinal loser (not the runner-up)
                        if match['loser'] not in semifinal_losers:
                            semifinal_losers.append(match['loser'])
        
        # Play 3rd place match if we have exactly 2 semifinal losers
        if len(semifinal_losers) == 2:
            s1 = semifinal_losers[0]
            s2 = semifinal_losers[1]
            
            score1, score2, _ = self.play_match(s1, s2)
            result = self.determine_winner(score1, score2, s1, s2)
            self.third_place = result['winner']
        elif len(semifinal_losers) == 1:
            # Only one semifinal loser available
            self.third_place = semifinal_losers[0]
        else:
            # If we can't determine 3rd place, find the best performing non-champion/runner-up
            strategy_performance = {}
            for s in self.strategies:
                if not isinstance(s, ByeStrategy) and s != self.champion and s != self.runner_up:
                    # Count wins
                    wins = 0
                    for match in self.match_history:
                        if match.get('winner') == s.name:
                            wins += 1
                    strategy_performance[s] = wins
            
            if strategy_performance:
                self.third_place = max(strategy_performance.items(), key=lambda x: x[1])[0]
        
        # SAFELY get names for return values
        champion_name = None
        if self.champion is not None:
            if hasattr(self.champion, 'name'):
                champion_name = self.champion.name
            else:
                champion_name = str(self.champion)
        
        runner_up_name = None
        if self.runner_up is not None:
            if hasattr(self.runner_up, 'name'):
                runner_up_name = self.runner_up.name
            else:
                runner_up_name = str(self.runner_up)
        
        third_place_name = None
        if self.third_place is not None:
            if hasattr(self.third_place, 'name'):
                third_place_name = self.third_place.name
            else:
                third_place_name = str(self.third_place)
        
        return {
            'champion': champion_name,
            'runner_up': runner_up_name,
            'third_place': third_place_name,
            'rounds': all_rounds,
            'match_history': self.match_history,
            'eliminated': eliminated,
            'total_rounds': round_number - 1
        }
    
    def run_double_elimination(self) -> Dict[str, Any]:
        """Run double elimination tournament"""
        # Create seeding
        seeded_strategies = self.create_seeding()
        
        # Initialize brackets - USE DOUBLE VERSION
        self.winners_bracket = self.create_initial_bracket_double(seeded_strategies)
        self.losers_bracket = []
        
        round_number = 1
        all_rounds = []
        self.match_history = []  # Reset match history
        
        # Main tournament loop
        while len(self.winners_bracket) > 1 or len(self.losers_bracket) > 1:
            round_results = {
                'round': round_number,
                'winners_matches': [],
                'losers_matches': []
            }
            
            # Play winners bracket matches
            new_winners_matches = []
            for match in self.winners_bracket:
                if not match['is_bye'] and not match['played']:
                    s1 = match['strategy1']
                    s2 = match['strategy2']
                    
                    # Skip if either strategy is None
                    if s1 is None or s2 is None:
                        continue
                    
                    score1, score2, history = self.play_match(s1, s2)
                    result = self.determine_winner(score1, score2, s1, s2)
                    
                    match['winner'] = result['winner']
                    match['loser'] = result['loser']
                    match['scores'] = (score1, score2)
                    match['played'] = True
                    
                    # SAFELY get names
                    s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                    s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                    winner_name = result['winner'].name if result['winner'] and hasattr(result['winner'], 'name') else str(result['winner'])
                    loser_name = result['loser'].name if result['loser'] and hasattr(result['loser'], 'name') else str(result['loser'])
                    
                    # Record match
                    match_result = {
                        'match_id': len(self.match_history),
                        'bracket': 'winners',
                        'round': round_number,
                        'strategy1': s1_name,
                        'strategy2': s2_name,
                        'winner': winner_name,
                        'loser': loser_name if result['loser'] else None,
                        'score1': score1,
                        'score2': score2,
                        'history': history
                    }
                    round_results['winners_matches'].append(match_result)
                    self.match_history.append(match_result)
                    
                    # Winner stays in winners bracket, loser goes to losers bracket
                    new_winners_matches.append(result['winner'])
                    if result['loser']:
                        self.losers_bracket.append(result['loser'])
            
            # Update winners bracket for next round
            if new_winners_matches:
                self.winners_bracket = []
                for i in range(0, len(new_winners_matches), 2):
                    if i + 1 < len(new_winners_matches):
                        match = {
                            'match_id': len(self.winners_bracket),
                            'strategy1': new_winners_matches[i],
                            'strategy2': new_winners_matches[i + 1],
                            'winner': None,
                            'loser': None,
                            'played': False,
                            'scores': (0, 0),
                            'is_bye': False
                        }
                        self.winners_bracket.append(match)
                    else:
                        # Odd number - this should not happen in double elimination
                        # But if it does, handle it
                        self.winners_bracket.append({
                            'match_id': len(self.winners_bracket),
                            'strategy1': new_winners_matches[i],
                            'strategy2': None,
                            'winner': new_winners_matches[i],
                            'loser': None,
                            'played': True,
                            'scores': (1000, 0),
                            'is_bye': True
                        })
            
            # Play losers bracket matches
            if self.losers_bracket and len(self.losers_bracket) >= 2:
                new_losers_matches = []
                
                # Sort losers bracket to ensure consistent pairing
                self.losers_bracket = [l for l in self.losers_bracket if l is not None]
                
                for i in range(0, len(self.losers_bracket), 2):
                    if i + 1 < len(self.losers_bracket):
                        s1 = self.losers_bracket[i]
                        s2 = self.losers_bracket[i + 1]
                        
                        score1, score2, history = self.play_match(s1, s2)
                        result = self.determine_winner(score1, score2, s1, s2)
                        
                        # SAFELY get names
                        s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                        s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                        winner_name = result['winner'].name if result['winner'] and hasattr(result['winner'], 'name') else str(result['winner'])
                        loser_name = result['loser'].name if result['loser'] and hasattr(result['loser'], 'name') else str(result['loser'])
                        
                        match_result = {
                            'match_id': len(self.match_history),
                            'bracket': 'losers',
                            'round': round_number,
                            'strategy1': s1_name,
                            'strategy2': s2_name,
                            'winner': winner_name,
                            'loser': loser_name if result['loser'] else None,
                            'score1': score1,
                            'score2': score2,
                            'history': history
                        }
                        round_results['losers_matches'].append(match_result)
                        self.match_history.append(match_result)
                        
                        # Winner stays in losers bracket
                        new_losers_matches.append(result['winner'])
                    else:
                        # Odd number - this one gets a bye
                        new_losers_matches.append(self.losers_bracket[i])
                
                self.losers_bracket = new_losers_matches
            
            all_rounds.append(round_results)
            round_number += 1
        
        # Determine winners bracket champion
        winners_champion = None
        if self.winners_bracket and len(self.winners_bracket) == 1:
            match = self.winners_bracket[0]
            if not match['played']:
                # Play winners bracket final
                if match['strategy1'] and match['strategy2']:
                    s1 = match['strategy1']
                    s2 = match['strategy2']
                    score1, score2, history = self.play_match(s1, s2)
                    result = self.determine_winner(score1, score2, s1, s2)
                    match['winner'] = result['winner']
                    match['loser'] = result['loser']
                    match['scores'] = (score1, score2)
                    match['played'] = True
                    
                    # SAFELY get names
                    s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                    s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                    winner_name = result['winner'].name if result['winner'] and hasattr(result['winner'], 'name') else str(result['winner'])
                    loser_name = result['loser'].name if result['loser'] and hasattr(result['loser'], 'name') else str(result['loser'])
                    
                    match_result = {
                        'match_id': len(self.match_history),
                        'bracket': 'winners_final',
                        'round': round_number,
                        'strategy1': s1_name,
                        'strategy2': s2_name,
                        'winner': winner_name,
                        'loser': loser_name,
                        'score1': score1,
                        'score2': score2,
                        'history': history
                    }
                    self.match_history.append(match_result)
                    winners_champion = result['winner']
                elif match['strategy1']:
                    winners_champion = match['strategy1']
                elif match['strategy2']:
                    winners_champion = match['strategy2']
            else:
                winners_champion = match['winner']
        
        # Determine losers bracket champion
        losers_champion = None
        if self.losers_bracket and len(self.losers_bracket) == 1:
            losers_champion = self.losers_bracket[0]
        elif self.losers_bracket and len(self.losers_bracket) > 1:
            # Need to play more losers bracket matches
            while len(self.losers_bracket) > 1:
                new_losers = []
                for i in range(0, len(self.losers_bracket), 2):
                    if i + 1 < len(self.losers_bracket):
                        s1 = self.losers_bracket[i]
                        s2 = self.losers_bracket[i + 1]
                        score1, score2, history = self.play_match(s1, s2)
                        result = self.determine_winner(score1, score2, s1, s2)
                        
                        # SAFELY get names
                        s1_name = s1.name if hasattr(s1, 'name') else str(s1)
                        s2_name = s2.name if hasattr(s2, 'name') else str(s2)
                        winner_name = result['winner'].name if result['winner'] and hasattr(result['winner'], 'name') else str(result['winner'])
                        loser_name = result['loser'].name if result['loser'] and hasattr(result['loser'], 'name') else str(result['loser'])
                        
                        match_result = {
                            'match_id': len(self.match_history),
                            'bracket': 'losers_final',
                            'round': round_number,
                            'strategy1': s1_name,
                            'strategy2': s2_name,
                            'winner': winner_name,
                            'loser': loser_name,
                            'score1': score1,
                            'score2': score2,
                            'history': history
                        }
                        self.match_history.append(match_result)
                        new_losers.append(result['winner'])
                    else:
                        new_losers.append(self.losers_bracket[i])
                self.losers_bracket = new_losers
                round_number += 1
            
            if self.losers_bracket:
                losers_champion = self.losers_bracket[0]
        
        # Grand finals
        if winners_champion and losers_champion:
            # First grand final match
            score1, score2, history = self.play_match(winners_champion, losers_champion)
            result = self.determine_winner(score1, score2, winners_champion, losers_champion)
            
            # SAFELY get names
            wc_name = winners_champion.name if hasattr(winners_champion, 'name') else str(winners_champion)
            lc_name = losers_champion.name if hasattr(losers_champion, 'name') else str(losers_champion)
            winner_name = result['winner'].name if result['winner'] and hasattr(result['winner'], 'name') else str(result['winner'])
            loser_name = result['loser'].name if result['loser'] and hasattr(result['loser'], 'name') else str(result['loser'])
            
            match_result = {
                'match_id': len(self.match_history),
                'bracket': 'grand_final',
                'round': round_number,
                'strategy1': wc_name,
                'strategy2': lc_name,
                'winner': winner_name,
                'loser': loser_name,
                'score1': score1,
                'score2': score2,
                'history': history
            }
            self.match_history.append(match_result)
            
            if result['winner'] == losers_champion:
                # Losers bracket champion won - need second match (true final)
                score1_2, score2_2, history_2 = self.play_match(winners_champion, losers_champion)
                result_2 = self.determine_winner(score1_2, score2_2, winners_champion, losers_champion)
                
                # SAFELY get names for second match
                winner_2_name = result_2['winner'].name if result_2['winner'] and hasattr(result_2['winner'], 'name') else str(result_2['winner'])
                loser_2_name = result_2['loser'].name if result_2['loser'] and hasattr(result_2['loser'], 'name') else str(result_2['loser'])
                
                match_result_2 = {
                    'match_id': len(self.match_history),
                    'bracket': 'grand_final_reset',
                    'round': round_number + 1,
                    'strategy1': wc_name,
                    'strategy2': lc_name,
                    'winner': winner_2_name,
                    'loser': loser_2_name,
                    'score1': score1_2,
                    'score2': score2_2,
                    'history': history_2
                }
                self.match_history.append(match_result_2)
                
                self.champion = result_2['winner']
                self.runner_up = result_2['loser']
            else:
                # Winners bracket champion won the first match
                self.champion = result['winner']
                self.runner_up = result['loser']
        elif winners_champion:
            self.champion = winners_champion
            self.runner_up = None
        elif losers_champion:
            self.champion = losers_champion
            self.runner_up = None
        
        # Determine 3rd place (loser of losers bracket finals)
        self.third_place = None
        if len(self.match_history) > 0:
            # Find the losers bracket final match
            losers_finals = [m for m in self.match_history if m.get('bracket') == 'losers_final']
            if losers_finals:
                # The loser of the losers bracket finals is 3rd place
                last_losers_final = losers_finals[-1]
                self.third_place = last_losers_final['loser']
            else:
                # If no explicit losers final, find the last losers bracket match before grand finals
                losers_matches = [m for m in self.match_history if m.get('bracket') == 'losers' and m.get('round', 0) > 0]
                if losers_matches:
                    # Sort by round and get the loser of the last one
                    losers_matches.sort(key=lambda x: x.get('round', 0))
                    last_losers_match = losers_matches[-1]
                    self.third_place = last_losers_match['loser']
        
        # SAFELY get names for return values
        champion_name = None
        if self.champion is not None:
            if hasattr(self.champion, 'name'):
                champion_name = self.champion.name
            else:
                champion_name = str(self.champion)
        
        runner_up_name = None
        if self.runner_up is not None:
            if hasattr(self.runner_up, 'name'):
                runner_up_name = self.runner_up.name
            else:
                runner_up_name = str(self.runner_up)
        
        third_place_name = None
        if self.third_place is not None:
            if hasattr(self.third_place, 'name'):
                third_place_name = self.third_place.name
            else:
                third_place_name = str(self.third_place)
        
        return {
            'champion': champion_name,
            'runner_up': runner_up_name,
            'third_place': third_place_name,
            'match_history': self.match_history,
            'rounds': all_rounds,
            'total_rounds': round_number - 1
        }
    
    def run_tournament(self) -> Dict[str, Any]:
        """Run the tournament based on bracket type"""
        if self.bracket_type == "single":
            return self.run_single_elimination()
        elif self.bracket_type == "double":
            return self.run_double_elimination()
        else:
            raise ValueError(f"Unknown bracket type: {self.bracket_type}")
    
    def get_bracket_visualization_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for bracket visualization"""
        if self.bracket_type == "single":
            return self._get_single_elim_bracket_data(results)
        else:
            return self._get_double_elim_bracket_data(results)
    
    def _get_single_elim_bracket_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare single elimination bracket data"""
        bracket_data = {
            'type': 'single',
            'rounds': [],
            'champion': results.get('champion'),
            'runner_up': results.get('runner_up'),
            'third_place': results.get('third_place')
        }
        
        # Convert match data to the format expected by visualizer
        max_round = max(m['round'] for m in results.get('match_history', [])) if results.get('match_history') else 0
        
        for round_num in range(1, max_round + 1):
            round_matches = [m for m in results.get('match_history', []) if m['round'] == round_num]
            
            round_info = {
                'round_number': round_num,
                'matches': []
            }
            
            for match in round_matches:
                match_info = {
                    'strategy1': match['strategy1'],
                    'strategy2': match['strategy2'],
                    'winner': match['winner'],
                    'score1': match['score1'],
                    'score2': match['score2'],
                    'played': True
                }
                round_info['matches'].append(match_info)
            
            bracket_data['rounds'].append(round_info)
        
        return bracket_data
    
    def _get_double_elim_bracket_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare double elimination bracket data"""
        bracket_data = {
            'type': 'double',
            'rounds': results.get('rounds', []),
            'champion': results.get('champion'),
            'runner_up': results.get('runner_up'),
            'third_place': results.get('third_place'),
            'match_history': results.get('match_history', [])
        }
        
        # If rounds are not in the expected format, reconstruct from match_history
        if (not bracket_data['rounds'] or len(bracket_data['rounds']) == 0) and results.get('match_history'):
            rounds_dict = {}
            for match in results['match_history']:
                # Group matches by round number
                round_num = match.get('round', 0)
                if round_num not in rounds_dict:
                    rounds_dict[round_num] = {
                        'round': round_num,
                        'winners_matches': [],
                        'losers_matches': []
                    }
                
                # Add match to appropriate bracket
                bracket_type = match.get('bracket', '')
                if 'winner' in bracket_type or 'final' in bracket_type:
                    if 'winners' in bracket_type:
                        rounds_dict[round_num]['winners_matches'].append(match)
                    else:
                        rounds_dict[round_num]['losers_matches'].append(match)
                elif match.get('bracket') == 'winners':
                    rounds_dict[round_num]['winners_matches'].append(match)
                elif match.get('bracket') == 'losers':
                    rounds_dict[round_num]['losers_matches'].append(match)
            
            # Convert to list and sort by round number
            if rounds_dict:
                bracket_data['rounds'] = [rounds_dict[r] for r in sorted(rounds_dict.keys())]
        
        return bracket_data
    
    def determine_third_place_double(self, results):
        """Determine 3rd place for double elimination"""
        # 3rd place is the loser of the losers bracket final
        losers_finals = [m for m in results.get('match_history', []) 
                        if m.get('bracket') == 'losers_final']
        
        if losers_finals:
            # Return the loser of the last losers final
            return losers_finals[-1]['loser']
        
        # If no explicit losers final, find the last losers bracket match before grand finals
        losers_matches = [m for m in results.get('match_history', []) 
                        if m.get('bracket') == 'losers' and not m.get('bracket', '').endswith('final')]
        if losers_matches:
            losers_matches.sort(key=lambda x: x.get('round', 0))
            return losers_matches[-1]['loser']
        
        return None