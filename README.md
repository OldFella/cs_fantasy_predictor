# CS Win Probability Estimator

A tool for estimating win probabilities for Counter-Strike matches and tournaments. Given two team names, it fetches their map statistics and world rankings, then simulates the match using a Bradley-Terry strength model combined with a Monte Carlo draft simulation.

## How it works

The estimator combines two signals to predict match outcomes:

- **Map strength** — each team's win rate per map, smoothed for small sample sizes and converted to a Bradley-Terry strength score
- **World ranking** — each team's global ranking points, blended with map strength via a configurable `alpha` parameter

For a given series format (BO1, BO3, BO5), the tool simulates thousands of drafts and map outcomes to produce a full probability distribution over all possible scorelines (e.g. 3-0, 3-1, 3-2 for a BO5).

## Architecture

The project follows a Ports & Adapters pattern to keep the data source and prediction model fully decoupled.

```
cs-win-predictor/
├── main.py                        # CLI entry point
└── src/
    ├── domain/
    │   ├── ports.py               # DataSource protocol (interface)
    │   ├── models.py              # BestOf, MatchOutcome data types
    │   └── estimator.py           # WinProbabilityEstimator abstract base
    ├── adapters/
    │   └── csapi.py               # Concrete adapter for csapi.de
    ├── models/
    │   └── elo.py                 # EloEstimator implementation
    └── tools/
        └── helpers.py             # Math utilities (Bradley-Terry, Monte Carlo)
```

**`DataSource` (port)** — defines the interface any data provider must implement. Swapping to a different API only requires a new adapter, nothing else changes.

**`WinProbabilityEstimator` (abstract base)** — defines `fit(datasource)`, `predict()`, and `predict_distribution()`. Any model that implements these three methods works as a drop-in replacement.

**`EloEstimator`** — the concrete implementation. Follows a scikit-learn style `fit`/`predict` interface. Caches team data and simulation results to avoid redundant API calls.

## Usage

```bash
python win_probabilities.py falcons natus-vincere --best-of 3
```

```
==============================
  Falcons vs Natus Vincere (BO3)
==============================
  Falcons win:        58.3%
  Natus Vincere win:  41.7%
──────────────────────────────
 outcome     n      p
      2-0  2032  0.203
      2-1  2471  0.247
      1-2  2826  0.283
      0-2  1671  0.267
==============================
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--best-of` | Series format: 1, 3, or 5 | `3` |
| `--alpha` | Blend weight for ranking vs map stats | `0.3` |
| `--tau` | Draft simulation temperature | `0.2` |
| `--n-simulations` | Number of Monte Carlo iterations | `10000` |
| `--no-rankings` | Disable ranking signal, use map stats only | `False` |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```