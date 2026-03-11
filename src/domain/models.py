import pandas as pd
from dataclasses import dataclass
from typing import Literal


BestOf = Literal[1, 3, 5]

@dataclass
class MatchOutcome:
    outcome: list[str]   # ["2-0", "2-1", "1-2", "0-2"]
    p: list[float]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({'outcome': self.outcome, 'p': self.p})
    
