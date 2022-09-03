import pandas as pd
from pathlib import Path
import sys
class ETL:
    def __init__(self):
        pass
    def drop_columns(self, df):
        df.drop(columns='Unnamed: 0', inplace=True)
        return df
    def rename(self, df):
        headers = {}
        df = df.rename(str.lower, axis='columns')
        return df
    def map(self, df):
        unique_vals = df['type'].unique()
        df['type'].replace(to_replace=unique_vals,
        value = list(range(len(unique_vals))),
        inplace=True)
        return df
    def fit(self, FILE_PATH):
        df = pd.read_csv(f'/usr/src/datasets/{FILE_PATH}', index_col=[0])
        #df = self.drop_columns(df)
        df = self.rename(df)
        df = self.map(df)
        df.reset_index(inplace=True)
        print(df.columns)
        filepath_main = Path('/usr/src/etled-data/etled-data.csv')
        filepath_twenty = Path('/usr/src/etled-data/lower_twenty.csv')
        filepath_forty= Path('/usr/src/etled-data/lower_forty.csv')
        filepath_sixty= Path('/usr/src/etled-data/lower_sixty.csv')
        filepath_eighty= Path('/usr/src/etled-data/lower_eighty.csv')
        filepath_main.parent.mkdir(parents=True, exist_ok=True)
        filepath_twenty.parent.mkdir(parents=True, exist_ok=True)
        filepath_forty.parent.mkdir(parents=True, exist_ok=True)
        filepath_sixty.parent.mkdir(parents=True, exist_ok=True)
        filepath_eighty.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath_main, index=False)
        df.loc[(df['Age'] > 0) & (df['Age'] <= 20)].to_csv(filepath_twenty, index=False)
        df.loc[(df['Age'] > 20) & (df['Age'] <= 40)].to_csv(filepath_forty, index=False)
        df.loc[(df['Age'] > 40) & (df['Age'] <= 60)].to_csv(filepath_sixty, index=False)
        df.loc[(df['Age'] > 60) & (df['Age'] <= 80)].to_csv(filepath_eighty, index=False)
        return df
    
if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    etl = ETL()
    df = etl.fit(raw_data_path)
    