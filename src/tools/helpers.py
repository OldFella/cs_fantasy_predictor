import numpy as np
import pandas as pd

from src.domain.models import BestOf


# --- Maths ---

def softmax_t(x, tau):
    """
    Compute softmax with temperature parameter tau.
    Lower tau makes the distribution more peaked (deterministic),
    higher tau makes it more uniform (random).
    
    Args:
        x (np.array): Input array
        tau (float): Temperature parameter
    Returns:
        np.array: Softmax probabilities
    """
    e = np.exp(x / tau)
    return e / e.sum()


# --- Strength ---

def strength_maps(n_wins, n_games, c=5, K=400):
    """
    Compute Bradley-Terry strength scores from map win/loss records.
    Applies additive smoothing to handle small sample sizes, pulling
    win rates toward 0.5 for teams with few games played.
    Teams with 0 games played are assigned a strength of 0 (neutral).
    
    Args:
        n_wins (np.array): Number of wins per map
        n_games (np.array): Number of games played per map
        c (int): Smoothing prior. Higher c pulls win rate harder toward 0.5
        K (int): Elo scale factor. Strength difference of K = 10x win rate advantage
    Returns:
        np.array: Strength scores per map
    """
    penalty = c / (n_games + 1)

    p_smoothed = (n_wins+ c)/(n_games + ((2* c) + penalty)) 
    
    strength = K * np.log10(p_smoothed/(1-p_smoothed))
    strength[n_games == 0] = -400
    
    return strength


def strength(score, K=400):
    """
    Convert a win rate or ranking score to a Bradley-Terry strength value
    using a log-odds transformation scaled by K.
    Equivalent to an Elo-style rating where a difference of K corresponds
    to a 10x win rate advantage.
    
    Args:
        score (float or np.array): Win rate or ranking score in (0, 1)
        K (int): Elo scale factor
    Returns:
        float or np.array: Strength score
    """
    return K * np.log10(score / (1 - score))


def bradley_terry(R_i, R_j, base=10, K=400):
    return 1/(1+base**((R_j - R_i)/K))

# --- Simulation ---

def simulate_series(p_win_array, format='bo3', tau=0.2):
    """
    Simulate a single CS bo1/bo3/bo5 draft and return the win probabilities
    of the picked maps. Ban and pick decisions are made using a softmax over
    per-map win probabilities with temperature tau.
    
    Draft format:
        bo1: 6 bans, 1 decider
        bo3: 2 bans, 2 picks, 2 bans, 1 decider
        bo5: 2 bans, 4 picks, 1 decider

    Args:
        p_win_array (np.array): Per-map win probabilities for team 1
        format (str): Series format, one of 'bo1', 'bo3', 'bo5'
        tau (float): Softmax temperature. Lower = more deterministic draft
    Returns:
        np.array: Win probabilities for the picked maps
    """
    p = p_win_array.copy()
    picks = []

    def do_ban(worst=True):
        probs = softmax_t(1 - p, tau) if worst else softmax_t(p, tau)
        idx = np.random.choice(len(p), p=probs)
        return idx

    def do_pick(best=True):
        probs = softmax_t(p, tau) if best else softmax_t(1 - p, tau)
        idx = np.random.choice(len(p), p=probs)
        return idx

    def remove(idx):
        nonlocal p
        p = np.delete(p, idx)

    def pick_and_remove(best=True):
        idx = do_pick(best)
        val = p[idx]
        remove(idx)
        picks.append(val)

    # First ban rotation (shared across all formats)
    remove(do_ban(worst=True))
    remove(do_ban(worst=False))

    if format == 'bo1':
        remove(do_ban(worst=True))
        remove(do_ban(worst=False))
        remove(do_ban(worst=True))
        remove(do_ban(worst=False))
    elif format == 'bo3':
        pick_and_remove(best=True)
        pick_and_remove(best=False)
        remove(do_ban(worst=False))
        remove(do_ban(worst=True))
    elif format == 'bo5':
        pick_and_remove(best=True)
        pick_and_remove(best=False)
        pick_and_remove(best=True)
        pick_and_remove(best=False)

    # Decider
    picks.append(p[0])
    return np.array(picks)


def mc_sim(p_win_array, format='bo3', N=10000, tau=0.2):
    """
    Run a Monte Carlo simulation of a CS series to estimate outcome probabilities.
    Each iteration simulates the full draft and then simulates each map result
    independently based on the per-map win probability.
    
    Args:
        p_win_array (np.array): Per-map win probabilities for team 1
        format (str): Series format, one of 'bo1', 'bo3', 'bo5'
        N (int): Number of simulations
        tau (float): Softmax temperature for draft simulation
    Returns:
        dict: Outcome counts keyed by score (e.g. {2: 4025, 1: 2891, -1: 1728, -2: 1356})
              Positive scores are wins for team 1, negative are wins for team 2
    """
    maps_to_win = {'bo1': 1, 'bo3': 2, 'bo5': 3}
    target = maps_to_win[format]
    outcomes = {}
    for _ in range(N):
        picks = simulate_series(p_win_array, format=format, tau=tau)
        score_a = 0
        score_b = 0
        for p_win in picks:
            roll = np.random.random()
            if roll < p_win:
                score_a += 1 
            else:
                score_b -= 1
            if abs(score_a) == target or abs(score_b) == target:
                break
        score = score_a + score_b
        outcomes[score] = outcomes.get(score, 0) + 1
    return outcomes

def outcome_to_df(outcomes: dict, best_of: BestOf) -> pd.DataFrame:
    maps_to_win = {1: 1, 3: 2, 5: 3}
    target = maps_to_win[best_of]

    def score_to_label(score: int) -> str:
        if score > 0:
            return f"{target}-{target - score}"
        else:
            return f"{target + score}-{target}"

    outcomes_sub = {score_to_label(k): v for k, v in outcomes.items()}
    df = pd.DataFrame({
        'outcome': list(outcomes_sub.keys()),
        'n': list(outcomes_sub.values())
    })
    df['p'] = df['n'] / df['n'].sum()
    return df