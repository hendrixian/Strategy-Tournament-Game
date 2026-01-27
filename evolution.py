"""
Evolutionary dynamics and Evolutionarily Stable Strategies (ESS) analysis.
"""
import numpy as np
from typing import List, Dict, Tuple, Any
from strategies import Strategy
from payoff import PayoffMatrix

class EvolutionaryDynamics:
    """Simulate evolutionary dynamics of strategies"""
    
    def __init__(self, strategies: List[Strategy], payoff_matrix: PayoffMatrix):
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix
        self.strategy_names = [s.name for s in strategies]
        self.n_strategies = len(strategies)
    
    def compute_fitness_matrix(self, rounds_per_match: int = 10) -> np.ndarray:
        """
        Compute fitness matrix F where F[i,j] is payoff of strategy i against j.
        
        Returns:
            n x n numpy array of average payoffs
        """
        F = np.zeros((self.n_strategies, self.n_strategies))
        
        for i in range(self.n_strategies):
            for j in range(self.n_strategies):
                s1 = self.strategies[i]
                s2 = self.strategies[j]
                
                # Play match
                total_score = 0
                history = []
                
                s1.reset()
                s2.reset()
                
                for _ in range(rounds_per_match):
                    move1 = s1.decide(history, s2.name)
                    move2 = s2.decide([(h[1], h[0]) for h in history], s1.name)
                    
                    payoff1, _ = self.payoff_matrix.get_payoff(move1, move2)
                    total_score += payoff1
                    
                    history.append((move1, move2))
                
                F[i, j] = total_score / rounds_per_match
        
        return F
    
    def replicator_dynamics(self, initial_pop: np.ndarray, 
                          generations: int = 100,
                          mutation_rate: float = 0.01) -> Tuple[np.ndarray, List]:
        """
        Simulate replicator dynamics.
        
        Args:
            initial_pop: Initial population frequencies (should sum to 1)
            generations: Number of generations to simulate
            mutation_rate: Rate of random mutation
        
        Returns:
            Tuple of (population_history, avg_fitness_history)
        """
        # Ensure initial population sums to 1
        initial_pop = initial_pop / initial_pop.sum()
        
        # Compute fitness matrix
        F = self.compute_fitness_matrix()
        
        # Initialize population history
        pop_history = np.zeros((generations, self.n_strategies))
        avg_fitness_history = []
        
        current_pop = initial_pop.copy()
        
        for gen in range(generations):
            # Store current population
            pop_history[gen] = current_pop
            
            # Calculate fitness for each strategy
            fitness = F @ current_pop  # Weighted average payoff against population
            
            # Calculate average fitness
            avg_fitness = current_pop @ fitness
            avg_fitness_history.append(avg_fitness)
            
            # Replicator equation: dx_i/dt = x_i * (f_i - φ)
            # Discrete version: x_i' = x_i * (f_i / φ)
            if avg_fitness > 0:
                new_pop = current_pop * fitness / avg_fitness
            else:
                new_pop = current_pop
            
            # Apply mutation (small random changes)
            if mutation_rate > 0:
                mutation = np.random.normal(0, mutation_rate, self.n_strategies)
                new_pop = new_pop + mutation
                new_pop = np.clip(new_pop, 0, 1)  # Ensure non-negative
                new_pop = new_pop / new_pop.sum()  # Renormalize
            
            # Update population
            current_pop = new_pop
        
        return pop_history, avg_fitness_history
    
    def is_ess(self, strategy_index: int, epsilon: float = 0.01) -> Dict[str, Any]:
        """
        Check if a strategy is Evolutionarily Stable.
        
        A strategy s is ESS if for all alternative strategies t ≠ s:
        1. Either E(s,s) > E(t,s), or
        2. E(s,s) = E(t,s) and E(s,t) > E(t,t)
        
        Returns:
            Dictionary with ESS analysis
        """
        F = self.compute_fitness_matrix()
        s = strategy_index
        
        conditions = []
        is_ess = True
        
        for t in range(self.n_strategies):
            if t == s:
                continue
            
            # Condition 1: E(s,s) > E(t,s)
            if F[s, s] > F[t, s]:
                conditions.append(f"F({s},{s}) = {F[s,s]:.3f} > F({t},{s}) = {F[t,s]:.3f}")
                continue
            
            # Condition 2: E(s,s) = E(t,s) and E(s,t) > E(t,t)
            if abs(F[s, s] - F[t, s]) < epsilon:
                if F[s, t] > F[t, t]:
                    conditions.append(f"F({s},{s}) = F({t},{s}) and F({s},{t}) = {F[s,t]:.3f} > F({t},{t}) = {F[t,t]:.3f}")
                    continue
            
            # Neither condition holds
            is_ess = False
            conditions.append(f"Strategy {self.strategy_names[t]} can invade")
        
        return {
            'strategy': self.strategy_names[strategy_index],
            'is_ess': is_ess,
            'conditions': conditions,
            'fitness_against_self': F[strategy_index, strategy_index]
        }
    
    def find_all_ess(self) -> List[Dict[str, Any]]:
        """Find all Evolutionarily Stable Strategies"""
        ess_results = []
        
        for i in range(self.n_strategies):
            result = self.is_ess(i)
            ess_results.append(result)
        
        return ess_results
    
    def simulate_evolutionary_tournament(self, initial_counts: Dict[str, int],
                                       generations: int = 50,
                                       carry_capacity: int = 100) -> Dict[str, Any]:
        """
        Simulate evolutionary tournament with discrete individuals.
        
        Args:
            initial_counts: Initial count of each strategy
            generations: Number of generations
            carry_capacity: Maximum population size
        
        Returns:
            Simulation results
        """
        # Initialize population
        population = initial_counts.copy()
        population_history = {name: [count] for name, count in initial_counts.items()}
        
        # Get strategy mapping
        strategy_map = {s.name: s for s in self.strategies}
        
        for gen in range(1, generations):
            # Calculate total population
            total_pop = sum(population.values())
            
            if total_pop == 0:
                # Population extinct
                for name in population.keys():
                    population_history[name].append(0)
                continue
            
            # Calculate fitness through random interactions
            fitness_scores = {}
            match_counts = {name: 0 for name in population.keys()}
            
            # Create list of individuals
            individuals = []
            for name, count in population.items():
                individuals.extend([name] * count)
            
            # Shuffle and have random interactions
            np.random.shuffle(individuals)
            n_individuals = len(individuals)
            
            # Each individual interacts with k others
            k = min(5, n_individuals - 1)
            
            for idx, name1 in enumerate(individuals):
                # Select random opponents
                opponent_indices = np.random.choice([i for i in range(n_individuals) if i != idx],
                                                  size=min(k, n_individuals-1), replace=False)
                
                for opp_idx in opponent_indices:
                    name2 = individuals[opp_idx]
                    
                    # Play one round
                    s1 = strategy_map[name1]
                    s2 = strategy_map[name2]
                    
                    move1 = s1.decide([], name2)
                    move2 = s2.decide([], name1)
                    
                    payoff1, _ = self.payoff_matrix.get_payoff(move1, move2)
                    
                    # Update fitness
                    if name1 not in fitness_scores:
                        fitness_scores[name1] = 0
                    fitness_scores[name1] += payoff1
                    match_counts[name1] += 1
            
            # Calculate average fitness
            avg_fitness = {}
            for name in population.keys():
                if match_counts[name] > 0:
                    avg_fitness[name] = fitness_scores.get(name, 0) / match_counts[name]
                else:
                    avg_fitness[name] = 0
            
            # Reproduction based on fitness
            new_population = {}
            total_fitness = sum(avg_fitness.values())
            
            if total_fitness > 0:
                for name, count in population.items():
                    # Fitness-proportional reproduction
                    if total_fitness > 0:
                        expected_offspring = count * (1 + avg_fitness[name] / max(avg_fitness.values()))
                    else:
                        expected_offspring = count
                    
                    # Add some randomness
                    offspring = int(np.random.poisson(expected_offspring))
                    
                    # Apply carrying capacity
                    if sum(new_population.values()) + offspring > carry_capacity:
                        offspring = max(0, carry_capacity - sum(new_population.values()))
                    
                    new_population[name] = offspring
            else:
                # If all fitness is zero, maintain current population
                new_population = population.copy()
            
            # Update population
            population = new_population
            
            # Record history
            for name in population_history.keys():
                population_history[name].append(population.get(name, 0))
        
        return {
            'population_history': population_history,
            'final_population': population,
            'generations': generations
        }