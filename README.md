# Strategy Tournament Game: An Interactive Game Theory Simulator

![Game Theory](https://img.shields.io/badge/Game-Theory-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Web-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive simulator that demonstrates classic game theory concepts through tournaments between different decision-making strategies in repeated games like the Prisoner's Dilemma.

## Overview

This project allows you to experiment with game theory concepts by pitting different strategies against each other in classic game scenarios. Watch as strategies compete, evolve, and demonstrate fundamental game theory principles like Nash Equilibrium, dominant strategies, and evolutionarily stable strategies.

## Features

### **Interactive Game Simulations**
- **Multiple Game Types**: Prisoner's Dilemma, Snowdrift Game, Stag Hunt, Matching Pennies
- **Customizable Parameters**: Adjust payoff matrices, noise levels, and match lengths
- **Real-time Visualization**: Watch matches unfold with interactive charts

###  **Strategy Library**
- **Always Cooperate**: Unconditionally cooperative
- **Always Defect**: Unconditionally competitive  
- **Tit for Tat**: Starts with cooperation, then mirrors opponent's last move
- **Tit for Two Tats**: Only defects after two consecutive defections
- **Grudger**: Cooperates until first betrayal, then always defects
- **Pavlov**: Win-Stay, Lose-Shift strategy
- **Random**: Randomly chooses between cooperate/defect
- **Adaptive TFT**: Adjusts based on opponent's behavior patterns

###  **Analysis Tools**
- **Tournament Rankings**: Round-robin competitions with detailed statistics
- **Cooperation Heatmaps**: Visualize cooperation patterns between strategies
- **Evolutionary Dynamics**: Simulate population evolution over generations
- **ESS Analysis**: Identify Evolutionarily Stable Strategies
- **Nash Equilibrium Calculator**: Automatic game analysis

### **User Interface**
- **Streamlit Web App**: No installation required to run the web interface
- **Interactive Controls**: Sliders, dropdowns, and checkboxes for all parameters
- **Responsive Design**: Works on desktop and mobile browsers
- **Educational Content**: Built-in explanations of game theory concepts

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hendrixian/Strategy-Tournament-Game.git
cd strategy-tournament-game
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

#### Option 2: Command Line Interface
```bash
python main.py
```

## Project Structure

```
strategy_tournament/
├── app.py              # Streamlit web application
├── main.py             # Command-line interface
├── strategies.py       # Strategy implementations (Always Cooperate, Tit for Tat,etc.)
├── payoff.py           # Game payoff matrices and analysis
├── tournament.py       # Tournament management and match scheduling
├── evolution.py        # Evolutionary dynamics and ESS analysis
├── analysis.py         # Statistical analysis and visualization
├── requirements.txt    # Python dependencies
└── README.md         
```

## What You Can Experiment With

### 1. **Basic Tournament**
Run a round-robin tournament to see which strategies perform best:
- Configure payoff matrices (T, R, P, S values)
- Adjust noise levels (probability of mistakes)
- Set number of rounds per match
- View cooperation rates and scores

### 2. **Evolutionary Simulation**
Watch strategies evolve over generations:
- Set initial population distributions
- Adjust mutation rates
- Observe population dynamics
- Identify Evolutionarily Stable Strategies (ESS)

### 3. **Single Match Analysis**
Study individual matchups in detail:
- Track move-by-move decisions
- Analyze cooperation patterns
- Understand strategy behavior

### 4. **Game Theory Concepts**
Interactive demonstrations of:
- **Prisoner's Dilemma**: Why rational players might not cooperate
- **Nash Equilibrium**: Stable strategy combinations
- **Dominant Strategies**: Strategies that are always best
- **ESS**: Strategies that can't be invaded by alternatives
- **Iterated Games**: How repetition changes outcomes

## Educational Value

This simulator is perfect for:
- **Students**: Visualize abstract game theory concepts
- **Teachers**: Classroom demonstrations and assignments
- **Researchers**: Test hypotheses about strategic behavior
- **Enthusiasts**: Explore the fascinating world of strategic interactions

### Concepts Demonstrated:
- Matrix games and payoff matrices
- Dominance and best responses
- Nash equilibrium (pure and mixed)
- Prisoner's Dilemma and its variants
- Strategic moves (reciprocity, retaliation, forgiveness)
- Utility/payoff maximization
- Evolutionarily Stable Strategies (ESS)
- Repeated game strategies

## Technical Details

### Dependencies
- **NumPy**: Numerical computations and matrix operations
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation and analysis
- **Streamlit**: Interactive web application framework

## Example Results

### Typical Prisoner's Dilemma Findings
- **Tit for Tat** often wins tournaments by balancing cooperation and retaliation
- **Always Defect** does well against naive cooperators but fails in diverse populations
- **Grudger** struggles against strategies that test cooperation
- Noise (mistakes) can dramatically change outcomes

### Evolutionarily Stable Strategies
- In standard Prisoner's Dilemma, **Always Defect** is often an ESS
- With repeated interactions, **Tit for Tat** can become stable
- Population composition affects which strategies survive

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Inspired by Robert Axelrod's famous Prisoner's Dilemma tournaments
- Based on concepts from evolutionary game theory
- Built with amazing open-source tools (Python, Streamlit, Matplotlib)
