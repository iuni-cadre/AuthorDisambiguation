import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

if __name__ == "__main__":

    disambiguated_file = sys.argv[1]  # "data/disambiguated-authors.csv"
    original_file = sys.argv[2]  # "data/disambiguationBenchmarkLabels.csv"
    OUTPUT = sys.argv[3]

    #
    # Loading
    #
    # Load the predicted labels
    predicted_table = pd.read_csv(disambiguated_file)
    predicted_table = predicted_table.rename(
        columns={"_seq_no": "author_order", "paper_id": "WoSid"}
    )

    # Load the ground-truth labels
    gtruth_table = pd.read_csv(original_file)
    gtruth_table = gtruth_table.dropna()

    #
    # Preprocess the ground-truth table for merging
    #
    # Slice only those in the predicted
    wos_ids = predicted_table[["WoSid"]].drop_duplicates()
    gtruth_table = gtruth_table[np.isin(gtruth_table["WoSid"].values, wos_ids)]

    dflist = []
    for i, row in tqdm(
        gtruth_table[["WoSid", "author_order", "ID_researcher"]].iterrows()
    ):
        author_order = row["author_order"].split(";")
        author_ids = row["ID_researcher"].split(";")
        dg = pd.DataFrame(
            {
                "author_order": author_order,
                "ID_researcher": author_ids,
                "WoSid": row["WoSid"],
            }
        )
        dflist += [dg]
    gtruth_table = pd.concat(dflist, ignore_index=True)
    gtruth_table["author_order"] = gtruth_table["author_order"].astype(int)
    gtruth_table["ID_researcher"] = gtruth_table["ID_researcher"].astype(int)

    #
    # Merge the ground-truth and predicted labels by WoSid and author order
    #
    df = pd.merge(
        predicted_table, gtruth_table, on=["WoSid", "author_order"], how="left"
    )
    df = df.dropna()
    df["author_order"] = df["author_order"].astype(int)
    df["ID_researcher"] = df["ID_researcher"].astype(int)

    # Rename the columns for interpretability
    df = df[["disambiguated_author_id", "ID_researcher", "WoSid"]].rename(
        columns={
            "disambiguated_author_id": "predicted_author_id",
            "ID_researcher": "true_author_id",
        }
    )

    #
    # Save to file
    #
    df.to_csv(OUTPUT, index=False)
