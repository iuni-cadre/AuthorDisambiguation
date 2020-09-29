import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":

    name_count_file = sys.argv[1]
    general_name_list_file = sys.argv[2]

    df = pd.read_csv(name_count_file)
    df = df.rename(columns={"firstCombo": "first_name"})
    df["first_name"] = df["first_name"].str.replace("[^a-zA-Z ]", "")
    df = df.groupby("first_name")["paperCount"].agg("sum").sort_values().reset_index()
    df = df.loc[df.paperCount >= 1000]
    df[["first_name", "paperCount"]].to_csv(general_name_list_file, index=False)
