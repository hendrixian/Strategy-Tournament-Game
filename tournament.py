"""
Handle tournaments where strategies compete against each other.
"""
import random
from typing import List, Tuple, Dict, Any
import numpy as np
from strategies import Strategy
from payoff import PayoffMatrix

class Tournament:
    """Manages a tournament between multiple strategies"""
    
    def __init__(self, strategies: List[Strategy], payoff_matrix: PayoffMatrix,
                 rounds_per_match: int = 10, noise: float = 0.0):
        """
        Initialize a tournament.
        
        Args:
            strategies: List of strategy instances
            payoff_matrix: Payoff matrix for the game
            rounds_per_match: Number of rounds per pairwise match
            noise: Probability of a move being flipped (misimplementation)
        """
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix
        self.rounds_per_match = rounds_per_match
        self.noise = noise
        
        # Results storage
        self.scores = {s.name: 0.0 for s in strategies}
        self.match_results = []
        self.detailed_history = {s.name: [] for s in strategies}
    
    def apply_noise(self, move: str) -> str:
        """Apply noise to a move (with probability noise, flip the move)"""
        if random.random() < self.noise:
            return 'C' if move == 'D' else 'D'
        return move
    
  
    def play_match(self, strategy1: Strategy, strategy2: Strategy) -> Tuple[float, float, List]:
        """
        Play a match between two strategies.
        
        Returns:
            Tuple of (score1, score2, history)
        """
        # Reset strategies for a fresh match
        strategy1.reset()
        strategy2.reset()
        
        history = []
        move_history = []  # Separate list for strategy decisions (only moves)
        total_score1 = 0
        total_score2 = 0
        
        for round_num in range(self.rounds_per_match):
            # Get decisions using move_history (only moves, no payoffs)
            move1 = strategy1.decide(move_history, strategy2.name)
            move2 = strategy2.decide([(h[1], h[0]) for h in move_history], strategy1.name)
            
            # Apply noise
            move1 = self.apply_noise(move1)
            move2 = self.apply_noise(move2)
            
            # Get payoffs
            payoff1, payoff2 = self.payoff_matrix.get_payoff(move1, move2)
            
            # Update scores
            total_score1 += payoff1
            total_score2 += payoff2
            
            # Record full history (with payoffs)
            history.append((move1, move2, payoff1, payoff2))
            # Record move history (for strategy decisions)
            move_history.append((move1, move2))
        
        return total_score1, total_score2, history
    
    def run_round_robin(self) -> Dict[str, Any]:
        """Run a round-robin tournament where each strategy plays every other"""
        n = len(self.strategies)
        
        # Reset scores
        self.scores = {s.name: 0.0 for s in self.strategies}
        self.match_results = []
        self.detailed_history = {s.name: [] for s in self.strategies}
        
        # Play all pairs
        for i in range(n):
            for j in range(i + 1, n):
                s1 = self.strategies[i]
                s2 = self.strategies[j]
                
                # Play match
                score1, score2, history = self.play_match(s1, s2)
                
                # Update scores
                self.scores[s1.name] += score1
                self.scores[s2.name] += score2
                
                # Store match result
                self.match_results.append({
                    'strategy1': s1.name,
                    'strategy2': s2.name,
                    'score1': score1,
                    'score2': score2,
                    'history': history
                })
                
                # Store detailed history
                self.detailed_history[s1.name].extend([h[:2] + (h[2],) for h in history])
                self.detailed_history[s2.name].extend([(h[1], h[0], h[3]) for h in history])
        
        # Calculate average scores (normalize by number of opponents)
        avg_scores = {}
        for name, score in self.scores.items():
            avg_scores[name] = score / (n - 1) if n > 1 else score
        
        # Sort strategies by score
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'scores': avg_scores,
            'rankings': sorted_scores,
            'match_results': self.match_results,
            'detailed_history': self.detailed_history
        }
    
    def run_evolutionary_round(self, population: Dict[str, int]) -> Dict[str, float]:
        """
        Run one round of evolutionary dynamics.
        
        Args:
            population: Dictionary mapping strategy names to counts
        
        Returns:
            New population frequencies
        """
        strategies = []
        for name, count in population.items():
            # Find strategy by name
            strategy = next((s for s in self.strategies if s.name == name), None)
            if strategy:
                strategies.extend([strategy] * count)
        
        # Shuffle the population
        random.shuffle(strategies)
        n = len(strategies)
        
        # Play random matches (each strategy plays against k others)
        k = min(5, n-1)  # Each strategy plays against k others
        scores = {s.name: 0.0 for s in self.strategies}
        match_counts = {s.name: 0 for s in self.strategies}
        
        # Create indices list
        indices = list(range(n))
        
        for i in range(n):
            s1 = strategies[i]
            
            # Randomly select k opponents (excluding self)
            possible_opponents = indices[:i] + indices[i+1:]
            if len(possible_opponents) > k:
                opponents = random.sample(possible_opponents, k)
            else:
                opponents = possible_opponents
            
            for j in opponents:
                s2 = strategies[j]
                
                # Play one round (or a few rounds)
                move1 = s1.decide([], s2.name)
                move2 = s2.decide([], s1.name)
                
                # Apply noise
                move1 = self.apply_noise(move1)
                move2 = self.apply_noise(move2)
                
                # Get payoff
                payoff1, payoff2 = self.payoff_matrix.get_payoff(move1, move2)
                
                # Update scores
                scores[s1.name] += payoff1
                scores[s2.name] += payoff2
                match_counts[s1.name] += 1
                match_counts[s2.name] += 1
        
        # Calculate average fitness
        fitness = {}
        for name in scores.keys():
            if match_counts[name] > 0:
                fitness[name] = scores[name] / match_counts[name]
            else:
                fitness[name] = 0
        
        return fitness
    
    def get_cooperation_rates(self) -> Dict[str, float]:
        """Calculate cooperation rates for each strategy from detailed history"""
        cooperation_rates = {}
        
        for name, history in self.detailed_history.items():
            if not history:
                cooperation_rates[name] = 0.0
                continue
            
            coop_count = sum(1 for move in history if move[0] == 'C')
            cooperation_rates[name] = coop_count / len(history)
        
        return cooperation_rates