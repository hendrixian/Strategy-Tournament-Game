# Strategy Tournament Game: An Interactive Game Theory Simulator

![Game Theory](https://img.shields.io/badge/Game-Theory-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Web-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive game-theory sandbox for exploring how classic strategies behave across repeated 2-player games.


## Features
**Strategy Tournament Game** lets you run tournaments and simulations between well-known decision strategies such as:
- Always Cooperate
- Always Defect
- Tit for Tat
- Tit for Two Tats
- Grudger
- Pavlov (Win-Stay, Lose-Shift)
- Random
- Adaptive variants

You can analyze outcomes in several classic games:
- Prisoner’s Dilemma
- Snowdrift (Chicken)
- Stag Hunt
- Matching Pennies

## Main Capabilities
- **Round-robin tournaments** with per-strategy scoring and rankings.
- **Single match drill-downs** with round-by-round move/payoff history.
- **Game analysis** including pure/mixed equilibrium summaries.
- **Evolution simulations** to observe population shifts over generations.
- **Visual analytics** (heatmaps, trend plots, and strategy comparison views).


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
#### Web App link
`https://strategytournament.streamlit.app/`

## Project Structure

```
strategy_tournament/
├── app.py                     # Streamlit UI
├── main.py                    # CLI entry point
├── strategies.py              # Strategy implementations
├── payoff.py                  # Payoff matrices + game-theory analysis
├── tournament.py              # Tournament engine
├── evolution.py               # Evolutionary dynamics
├── analysis.py                # Plotting + analysis helpers
├── bracket.py                 # Bracket/tournament helpers
├── bracket_visualization.py   # Bracket visual outputs
├── decision_tree_analysis.py  # Decision-tree based analysis helpers
├── requirements.txt
└── README.md
```

## Typical experiments

1. **Tournament robustness**
   - Vary rounds per match and noise.
   - Compare whether reciprocal strategies remain competitive.

2. **Payoff sensitivity**
   - Adjust T/R/P/S in symmetric games.
   - Observe ranking changes and cooperation rates.

3. **Matching Pennies behavior**
   - Verify no pure Nash equilibrium appears.
   - Inspect mixed-strategy interpretation in analysis output.

4. **Evolution over generations**
   - Tune mutation rate and number of generations.
   - Track long-run survival of strategy families.

---

## Who this is for

- Students learning repeated games and strategic adaptation.
- Instructors demonstrating game-theory ideas interactively.
- Hobbyists experimenting with strategy ecosystems.
- Researchers needing a lightweight sandbox for quick comparisons.

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Inspired by Robert Axelrod's famous Prisoner's Dilemma tournaments
- Based on concepts from evolutionary game theory
- Built with amazing open-source tools (Python, Streamlit, Matplotlib)

