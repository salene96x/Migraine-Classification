import pandas as pd
from pathlib import Path
class ETL:
    def __init__(self):
        pass
    def rename(self, df):
        headers = {}
        df = df.rename(str.lower, axis='columns')
        return df
    def fit(self, FILE_PATH):
        df = pd.read_csv(f'/usr/src/datasets/{FILE_PATH}')
        df = self.rename(df)
        filepath = Path('/usr/src/etled-data/etled-data.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)  
        return df
    