"""
Define different decision-making strategies for game theory scenarios.
Each strategy is a function that takes history and returns a move (C or D).
"""
import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Constants for moves
COOPERATE = "C"
DEFECT = "D"

def extract_moves(history_entry: Tuple) -> Tuple[str, str]:
    """Extract moves from a history entry, handling different formats."""
    if len(history_entry) >= 4:
        # Format: (move1, move2, payoff1, payoff2)
        return history_entry[0], history_entry[1]
    elif len(history_entry) >= 2:
        # Format: (move1, move2)
        return history_entry[0], history_entry[1]
    else:
        # Unknown format
        return COOPERATE, COOPERATE
    
class Strategy:
    """Base class for all strategies"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def decide(self, history: List[Tuple[str, str]], opponent_name: str = "") -> str:
        """
        Make a decision based on game history.
        
        Args:
            history: List of tuples (my_move, opponent_move) from previous rounds
            opponent_name: Name of opponent strategy (for adaptive strategies)
        
        Returns:
            Either "C" (cooperate) or "D" (defect)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset(self):
        """Reset any internal state for a new match"""
        pass
    
    def __str__(self):
        return self.name

class AlwaysCooperate(Strategy):
    """Always chooses to cooperate"""
    def __init__(self):
        super().__init__("Always Cooperate", "Always chooses cooperation (C)")
    
    def decide(self, history: List[Tuple[str, str]], opponent_name: str = "") -> str:
        return COOPERATE

class AlwaysDefect(Strategy):
    """Always chooses to defect"""
    def __init__(self):
        super().__init__("Always Defect", "Always chooses defection (D)")
    
    def decide(self, history: List[Tuple[str, str]], opponent_name: str = "") -> str:
        return DEFECT

class TitForTat(Strategy):
    """Starts with cooperation, then copies opponent's last move"""
    def __init__(self):
        super().__init__("Tit for Tat", "Starts with C, then copies opponent's last move")
    
    def decide(self, history: List[Tuple], opponent_name: str = "") -> str:
        if not history:
            return COOPERATE
        
        # Extract opponent's last move from history
        last_round = history[-1]
        if len(last_round) >= 2:
            return last_round[1]  # Opponent's last move
        else:
            return COOPERATE  # Fallback
        
class TitForTwoTats(Strategy):
    """Only defects after two consecutive defections by opponent"""
    def __init__(self):
        super().__init__("Tit for Two Tats", "Defects only after two consecutive D's from opponent")
    
    def decide(self, history: List[Tuple], opponent_name: str = "") -> str:
        if len(history) < 2:
            return COOPERATE
        
        # Check last two rounds
        last_round1 = history[-1]
        last_round2 = history[-2]
        
        if (len(last_round1) >= 2 and len(last_round2) >= 2 and
            last_round1[1] == DEFECT and last_round2[1] == DEFECT):
            return DEFECT
        return COOPERATE

class RandomStrategy(Strategy):
    """Randomly chooses between cooperate and defect"""
    def __init__(self, cooperation_prob: float = 0.5):
        super().__init__("Random", f"Random choice with P(C)={cooperation_prob}")
        self.cooperation_prob = cooperation_prob
    
    def decide(self, history: List[Tuple[str, str]], opponent_name: str = "") -> str:
        return COOPERATE if random.random() < self.cooperation_prob else DEFECT

class Grudger(Strategy):
    """Cooperates until opponent defects once, then always defects"""
    def __init__(self):
        super().__init__("Grudger", "Cooperates until first D from opponent, then always D")
        self.grudging = False
    
    def decide(self, history: List[Tuple], opponent_name: str = "") -> str:
        if not history:
            self.grudging = False
            return COOPERATE
        
        # Check if opponent ever defected
        if not self.grudging:
            for last_round in history:
                if len(last_round) >= 2 and last_round[1] == DEFECT:
                    self.grudging = True
                    break
        
        return DEFECT if self.grudging else COOPERATE
    
    def reset(self):
        self.grudging = False

class Pavlov(Strategy):
    """
    Win-Stay, Lose-Shift strategy
    Repeats previous move if it was successful (got high payoff)
    Changes move if it was unsuccessful (got low payoff)
    """
    def __init__(self):
        super().__init__("Pavlov", "Win-Stay, Lose-Shift strategy")
        self.last_move = COOPERATE
    
    def decide(self, history: List[Tuple], opponent_name: str = "") -> str:
        if not history:
            self.last_move = COOPERATE
            return self.last_move
        
        # Extract moves from history (handle different formats)
        last_round = history[-1]
        if len(last_round) >= 2:
            # Extract moves (could be from (move1, move2) or (move1, move2, payoff1, payoff2))
            my_last = last_round[0]
            opp_last = last_round[1]
        else:
            # Fallback if history format is unexpected
            return self.last_move
        
        # For simplicity, assume "win" if both cooperated or both defected
        if my_last == opp_last:
            # Win - stay with same move
            return self.last_move
        else:
            # Lose - shift move
            self.last_move = DEFECT if self.last_move == COOPERATE else COOPERATE
            return self.last_move
    
    def reset(self):
        self.last_move = COOPERATE

class AdaptiveTFT(Strategy):
    """Tit for Tat but adapts to opponent patterns"""
    def __init__(self):
        super().__init__("Adaptive TFT", "TFT that adjusts based on opponent behavior")
    
    def decide(self, history: List[Tuple], opponent_name: str = "") -> str:
        if not history:
            return COOPERATE
        
        # Count opponent's cooperation rate
        coop_count = 0
        total_moves = 0
        
        for last_round in history:
            if len(last_round) >= 2:
                if last_round[1] == COOPERATE:  # Opponent's move
                    coop_count += 1
                total_moves += 1
        
        if total_moves == 0:
            return COOPERATE
        
        coop_rate = coop_count / total_moves
        
        # If opponent cooperates more than 70%, cooperate
        if coop_rate > 0.7:
            return COOPERATE
        # If opponent defects more than 60%, defect
        elif coop_rate < 0.4:
            return DEFECT
        # Otherwise, standard TFT
        else:
            last_round = history[-1]
            if len(last_round) >= 2:
                return last_round[1]  # Opponent's last move
            else:
                return COOPERATE

def get_all_strategies() -> List[Strategy]:
    """Return a list of all available strategies"""
    return [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        TitForTwoTats(),
        RandomStrategy(),
        Grudger(),
        Pavlov(),
        AdaptiveTFT(),
    ]

def get_strategy_by_name(name: str) -> Optional[Strategy]:
    """Get a strategy instance by its name"""
    for strategy in get_all_strategies():
        if strategy.name == name:
            return strategy
    return None