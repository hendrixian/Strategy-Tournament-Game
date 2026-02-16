"""
Define payoff matrices for different game theory scenarios.
"""
from typing import Dict, Tuple, Any
import numpy as np

class PayoffMatrix:
    """Represents a payoff matrix for a 2-player symmetric game"""
    
    # Common game matrices
    PRISONERS_DILEMMA = "Prisoner's Dilemma"
    SNOWDRIFT = "Snowdrift Game"
    STAG_HUNT = "Stag Hunt"
    MATCHING_PENNIES = "Matching Pennies"
    
    def __init__(self, game_type: str = PRISONERS_DILEMMA, **kwargs):
        """
        Initialize a payoff matrix.
        
        Args:
            game_type: Type of game (PRISONERS_DILEMMA, SNOWDRIFT, etc.)
            **kwargs: Custom payoffs if needed
        """
        self.game_type = game_type
        
        if game_type == self.PRISONERS_DILEMMA:
            # Standard Prisoner's Dilemma parameters
            # T > R > P > S, and 2R > T + S
            self.T = kwargs.get('T', 5)  # Temptation to defect
            self.R = kwargs.get('R', 3)  # Reward for mutual cooperation
            self.P = kwargs.get('P', 1)  # Punishment for mutual defection
            self.S = kwargs.get('S', 0)  # Sucker's payoff
            
            # Validate PD conditions
            if not (self.T > self.R > self.P > self.S and 2*self.R > self.T + self.S):
                raise ValueError("Payoffs don't satisfy Prisoner's Dilemma conditions")
                
        elif game_type == self.SNOWDRIFT:
            # Snowdrift/Chicken game: T > R > S > P
            self.T = kwargs.get('T', 3)  # Temptation to defect
            self.R = kwargs.get('R', 2)  # Reward for mutual cooperation
            self.S = kwargs.get('S', 1)  # Sucker's payoff
            self.P = kwargs.get('P', 0)  # Punishment for mutual defection
            
        elif game_type == self.STAG_HUNT:
            # Stag Hunt: R > T >= P > S
            self.R = kwargs.get('R', 3)  # Reward for mutual cooperation
            self.T = kwargs.get('T', 2)  # Temptation to defect
            self.P = kwargs.get('P', 1)  # Punishment for mutual defection
            self.S = kwargs.get('S', 0)  # Sucker's payoff
            
        elif game_type == self.MATCHING_PENNIES:
            # Zero-sum game
            self.win = kwargs.get('win', 1)
            self.lose = kwargs.get('lose', -1)
            
        else:
            raise ValueError(f"Unknown game type: {game_type}")
        
        # Create payoff lookup dictionary
        self._create_payoff_dict()
    
    def _create_payoff_dict(self):
        """Create payoff lookup dictionary based on game type"""
        if self.game_type == self.MATCHING_PENNIES:
            # Matching pennies is not symmetric in the same way
            self.payoffs = {
                ('C', 'C'): (self.lose, self.win),   # Player 1 loses, Player 2 wins
                ('C', 'D'): (self.win, self.lose),   # Player 1 wins, Player 2 loses
                ('D', 'C'): (self.win, self.lose),   # Player 1 wins, Player 2 loses
                ('D', 'D'): (self.lose, self.win),   # Player 1 loses, Player 2 wins
            }
        else:
            # For symmetric games
            self.payoffs = {
                ('C', 'C'): (self.R, self.R),
                ('C', 'D'): (self.S, self.T),
                ('D', 'C'): (self.T, self.S),
                ('D', 'D'): (self.P, self.P),
            }
    
    def get_payoff(self, move1: str, move2: str) -> Tuple[float, float]:
        """
        Get payoffs for two moves.
        
        Args:
            move1: Move of player 1 ('C' or 'D')
            move2: Move of player 2 ('C' or 'D')
        
        Returns:
            Tuple of (payoff_player1, payoff_player2)
        """
        return self.payoffs.get((move1, move2), (0, 0))
    
    def get_payoff_matrix(self) -> np.ndarray:
        """
        Return payoff matrix as a 2x2x2 numpy array.
        
        Returns:
            Array where matrix[0] is row player's payoffs and matrix[1] is column player's payoffs
            Rows correspond to row player's moves (C, D)
            Columns correspond to column player's moves (C, D)
        """
        matrix = np.zeros((2, 2, 2))  # 2 players x 2 moves x 2 moves
        
        moves = ['C', 'D']
        for i, m1 in enumerate(moves):
            for j, m2 in enumerate(moves):
                payoff = self.get_payoff(m1, m2)
                matrix[0, i, j] = payoff[0]  # Row player payoff
                matrix[1, i, j] = payoff[1]  # Column player payoff
        
        return matrix
    
    def analyze_nash_equilibrium(self) -> Dict[str, Any]:
        """
        Analyze the game for Nash equilibria.
        
        Returns:
            Dictionary with analysis results
        """
        matrix = self.get_payoff_matrix()
        row_payoffs = matrix[0]  # Payoffs for row player
        col_payoffs = matrix[1]  # Payoffs for column player
        
        moves = ['C', 'D']
        nash_equilibria = []
        
        # Check all pure strategy combinations
        for i, m1 in enumerate(moves):
            for j, m2 in enumerate(moves):
                # Check if row player wants to deviate
                row_best = True
                for k in range(2):
                    if row_payoffs[k, j] > row_payoffs[i, j]:
                        row_best = False
                        break
                
                # Check if column player wants to deviate
                col_best = True
                for k in range(2):
                    if col_payoffs[i, k] > col_payoffs[i, j]:
                        col_best = False
                        break
                
                if row_best and col_best:
                    nash_equilibria.append((m1, m2))
        
        if self.game_type == self.MATCHING_PENNIES:
                    payoffs = {
                        'win': getattr(self, 'win', 1),
                        'lose': getattr(self, 'lose', -1),
                    }
        else:
            payoffs = {
                'T': getattr(self, 'T', 0),
                'R': getattr(self, 'R', 0),
                'P': getattr(self, 'P', 0),
                'S': getattr(self, 'S', 0),
            }

        mixed_eq = None
        if self.game_type == self.MATCHING_PENNIES:
            # Standard mixed equilibrium for matching pennies
            mixed_eq = {'p_cooperate': 0.5, 'payoff': 0.0}
        else:
            try:
                denominator = self.T + self.S - self.R - self.P
                p = (self.S - self.P) / denominator
                if 0 <= p <= 1:
                    mixed_eq = {'p_cooperate': p, 'payoff': p*self.S + (1-p)*self.P}
            except (ZeroDivisionError, AttributeError):
                mixed_eq = None
                payoffs = {'T': self.T, 'R': self.R, 'P': self.P, 'S': self.S}
        
        return {
            'game_type': self.game_type,
            'payoffs': payoffs,            
            'pure_nash_equilibria': nash_equilibria,
            'mixed_equilibrium': mixed_eq,
            'dominant_strategy': self._find_dominant_strategy(row_payoffs, col_payoffs)
        }
    
    def _find_dominant_strategy(self, row_payoffs: np.ndarray, col_payoffs: np.ndarray) -> Dict[str, Any]:
        """Find dominant strategies for each player"""
        moves = ['C', 'D']
        row_dominant = []
        col_dominant = []
        
        # Check for row player
        for i, m1 in enumerate(moves):
            dominant = True
            for j, m2 in enumerate(moves):
                if i != j:
                    # Check if m1 dominates m2
                    if not all(row_payoffs[i, :] >= row_payoffs[j, :]):
                        dominant = False
                        break
            if dominant:
                row_dominant.append(m1)
        
        # Check for column player
        for i, m1 in enumerate(moves):
            dominant = True
            for j, m2 in enumerate(moves):
                if i != j:
                    # Check if m1 dominates m2
                    if not all(col_payoffs[:, i] >= col_payoffs[:, j]):
                        dominant = False
                        break
            if dominant:
                col_dominant.append(m1)
        
        return {'row': row_dominant, 'column': col_dominant}
    
    def __str__(self):
        """String representation of the payoff matrix"""
        if self.game_type == self.MATCHING_PENNIES:
            return f"Matching Pennies: Win={self.win}, Lose={self.lose}"
        else:
            return (f"{self.game_type}: T={self.T}, R={self.R}, P={self.P}, S={self.S}")