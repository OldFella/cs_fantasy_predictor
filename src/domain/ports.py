from typing import Protocol
from pandas import DataFrame


class DataSource(Protocol):
    def fetch_team_data(self, teamid: int)-> DataFrame: ...
    def fetch_rankings(self) -> DataFrame: ...
