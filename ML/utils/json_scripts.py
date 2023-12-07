import json

import pandas as pd


def convert_dataframe_to_json(df: pd.DataFrame | pd.Series) -> json:
    return df.sort_index().to_json(orient='index')


def convert_json_to_dataframe(json_file):
    d = json.loads(json_file)
    df = pd.DataFrame.from_dict(d, orient='index')
    df.index = df.index.astype('int')
    return df.sort_index()
