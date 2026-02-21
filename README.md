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
- Prisoner's Dilemma
- Snowdrift (Chicken)
- Stag Hunt
- Matching Pennies

## Main Capabilities
- Round-robin tournaments with per-strategy scoring and rankings.
- Single match drill-downs with round-by-round move/payoff history.
- Game analysis including pure/mixed equilibrium summaries.
- Evolution simulations to observe population shifts over generations.
- Visual analytics (heatmaps, trend plots, and strategy comparison views).

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

## Running the Application

### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`.

### Option 2: Command Line Interface
```bash
python main.py
```

Web app: `https://strategytournament.streamlit.app/`

## Running Simulation Scripts

Run simulation scripts from the project root with module syntax so imports resolve correctly.

### Multi-seed tournament run
```bash
python -m simulation.test_simulation --num-seeds 30 --start-seed 0 --rounds 50 --noise 0.2
```

### Plot latest results
```bash
python -m simulation.plot
```

Notes:
- Do not add a trailing `--` to the command unless you are forwarding extra args intentionally.
- Running `python simulation/test_simulation.py` directly can fail with import errors depending on working directory.

## Results Files

Tournament runs write files to `results/`:
- `*.txt`: detailed terminal output capture.
- `*.json`: aggregate metrics used by plotting.

`simulation.plot` expects JSON data with keys including `summary_rows` and `rank_samples`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'payoff'`
Use module execution from project root:
```bash
python -m simulation.test_simulation ...
```

### `KeyError: 'rank_samples'` when running plot
This means the selected JSON file is from an older output format that does not include `rank_samples`.
Generate a new JSON via:
```bash
python -m simulation.test_simulation --num-seeds 2 --start-seed 0 --rounds 10 --noise 0.2
```
Then run:
```bash
python -m simulation.plot
```

## Project Structure

```text
strategy/
|- app.py
|- main.py
|- strategies.py
|- payoff.py
|- tournament.py
|- evolution.py
|- analysis.py
|- bracket.py
|- bracket_visualization.py
|- decision_tree_analysis.py
|- simulation/
|  |- __init__.py
|  |- test_simulation.py
|  |- plot.py
|- results/
|- requirements.txt
`- README.md
```

## Typical Experiments

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

## Who This Is For

- Students learning repeated games and strategic adaptation.
- Instructors demonstrating game-theory ideas interactively.
- Hobbyists experimenting with strategy ecosystems.
- Researchers needing a lightweight sandbox for quick comparisons.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- Inspired by Robert Axelrod's famous Prisoner's Dilemma tournaments.
- Based on concepts from evolutionary game theory.
- Built with Python, Streamlit, and Matplotlib.
