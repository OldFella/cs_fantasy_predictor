from abc import ABC, abstractmethod
from src.domain.ports import DataSource
from src.domain.models import BestOf, MatchOutcome


class WinProbabilityEstimator(ABC):
    @abstractmethod
    def fit(self, datasource:DataSource) -> None:...
    
    @abstractmethod
    def predict(self, team_id_a:int, team_id_b:int, best_of:BestOf) ->float: ...

    @abstractmethod
    def predict_distribution(self, team_id_a:int, team_id_b:int, best_of:BestOf) -> MatchOutcome: ...
