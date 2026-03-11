from dataclasses import field
import numpy as np
import pandas as pd
from src.domain.estimators import WinProbabilityEstimator
from src.domain.models import BestOf, MatchOutcome
from src.domain.ports import DataSource

from src.tools.helpers import strength_maps, strength, bradley_terry, mc_sim, outcome_to_df


class EloEstimator(WinProbabilityEstimator):

    def __init__(self, alpha:float = 0.3, tau: float = 0.2, n_simulations: int = 10000, use_rankings = True):
        self.alpha = alpha
        self.tau = tau
        self.use_rankings = use_rankings
        self.n_simulations = n_simulations
        self._datasource: DataSource | None = None
        self._rankings: pd.DataFrame | None = None
        self._team_cache: dict[int,pd.DataFrame] = {}
        self._outcome_cache: dict[tuple[int,int,int], pd.DataFrame] = {}

    def fit(self, datasource: DataSource) -> None:
        self._datasource = datasource
        self._rankings = datasource.fetch_rankings()

    def predict(self, team_id_a: int, team_id_b: int, best_of: BestOf) -> float:
        outcome = self.predict_distribution(team_id_a, team_id_b, best_of)
        # return sum of win probabilities for team_a
        maps_to_win = {1: 1, 3: 2, 5: 3}
        target = maps_to_win[best_of]
        return outcome[outcome['outcome'].str.startswith(str(target))]['p'].sum()

    def predict_distribution(self, team_id_a: int, team_id_b: int, best_of: BestOf) -> MatchOutcome:
        key = (team_id_a, team_id_b, best_of)
        if key not in self._outcome_cache:
            self._simulate_outcomes(team_id_a, team_id_b, best_of)
        return self._outcome_cache[key]
        
    
    def _simulate_outcomes(self, team_id_a: int, team_id_b: int, best_of: BestOf) -> None:
        team_data_a = self._fetch_team_data(team_id_a)
        team_data_b = self._fetch_team_data(team_id_b)

        team_data_a['team'] = 1
        team_data_b['team'] = 2

        team_data = pd.concat([team_data_a,team_data_b])

        team_data['strength'] = strength_maps(np.array(team_data['n_wins']),np.array(team_data['n']), c = 5)

        p_win = []
        for map_id in team_data['id'].unique():
            t1 = team_data[(team_data['id'] == map_id) & (team_data['team'] == 1)]
            t2 = team_data[(team_data['id'] == map_id) & (team_data['team'] == 2)]
            p_win.append(bradley_terry(t1['strength'].item(), t2['strength'].item()))
        
        p_win = np.array(p_win)

        if self.use_rankings:
            ranking_a = self._rankings[self._rankings['id'] == team_id_a]['points'].item()
            ranking_b = self._rankings[self._rankings['id'] == team_id_b]['points'].item()

            p_combined = self.alpha * bradley_terry(ranking_a, ranking_b) + (1-self.alpha) * p_win
        else:
            p_combined = p_win

        outcomes = mc_sim(p_combined, format=f"bo{best_of}", N=self.n_simulations)
        outcomes = outcome_to_df(outcomes, best_of=best_of)

        key = (team_id_a, team_id_b, best_of)
        self._outcome_cache[key] = outcomes
    
    def _fetch_team_data(self, team_id:int) -> pd.DataFrame:
        if team_id not in self._team_cache:
            self._team_cache[team_id] = self._datasource.fetch_team_data(team_id)
        return self._team_cache[team_id]