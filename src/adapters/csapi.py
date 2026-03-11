import requests
from pandas import DataFrame

from src.domain.ports import DataSource


class CSApiDataSource(DataSource):
    base_url:str = "https://api.csapi.de"


    def _team_stats_url(self,teamid:int) -> str:
        return f"{self.base_url}/teams/{teamid}/stats"
    
    def _rankings_url(self) -> str:
        return f"{self.base_url}/rankings/"
    
    def _fetch(self, url:str) -> dict:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_team_data(self, teamid:int) -> DataFrame:
        data = DataFrame(self._fetch(self._team_stats_url(teamid)))
        return data[data['id'] != 0]

    def fetch_rankings(self)-> DataFrame:
        rankings = self._fetch(self._rankings_url())['rankings']
        return DataFrame(rankings)
    def fetch_team_id(self, name: str) -> int:
        response = self._fetch(f"{self.base_url}/teams/?name={name}")
        return response[0]['id'],response[0]['name'] 
