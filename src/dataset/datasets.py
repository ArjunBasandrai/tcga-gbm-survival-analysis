import pandas as pd

def load_df(path):
    df = pd.read_csv(path).T
    df.columns = df.iloc[1]
    df = df[2:]
    df = df.reset_index()
    df = df.rename(columns={"index": "patient_id"})
    df.columns.name = None
    return df