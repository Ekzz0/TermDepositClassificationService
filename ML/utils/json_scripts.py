import json

import pandas as pd


def convert_dataframe_to_json(df: pd.DataFrame | pd.Series) -> json:
    return df.sort_index().to_json(orient='index')


def convert_json_to_dataframe(json_file):
    # d = json.loads(json_file)
    #df = pd.DataFrame.from_dict(d, orient='index')
    df = pd.DataFrame(json_file)
    df.index = df['id']
    print(df.columns)
    return df.dropna().sort_index()
