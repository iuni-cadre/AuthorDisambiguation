import glob
import pathlib

import numpy as np
import pandas as pd
import utils
from scipy import sparse
from ScoringRule import ScoringRule

if __name__ == "__main__":

    GENERAL_NAME_LIST_FILE = snakemake.input["general_name_list"]
    root = pathlib.Path(snakemake.input["data_path"])
    block_name = snakemake.params["block_name"]
    threshold = float(snakemake.params["threshold"])
    OUTPUT = snakemake.output["output_file"]

    #
    # Load
    #
    general_name_list = pd.read_csv(GENERAL_NAME_LIST_FILE)["first_name"].values

    tables = {}
    for table_name in [
        "block_table",
        "citing_table",
        "grant_table",
        "name_paper_address",
        "paper_address",
        "paper_table",
        "name_table",
        "name_paper_table",
    ]:
        if "table" not in table_name:
            table_name_extended = table_name + "_table"
        else:
            table_name_extended = table_name
        dflist = [
            pd.read_csv(f, escapechar="\\")
            for f in glob.glob(str(root / table_name / block_name / "*"))
        ]
        if len(dflist) != 0:
            tables[table_name_extended] = pd.concat(dflist)

    #
    # Clustering
    #
    block_table = tables["block_table"]
    res_table_list = []
    maxcid = 0
    for bid in block_table["block_id"].drop_duplicates():
        block_tables = {}
        for k, df in tables.items():
            block_tables[k] = df[df["block_id"] == bid].copy()
        scoring_func = ScoringRule(general_name_list, **block_tables)
        W, author_paper_table = scoring_func.eval()

        if isinstance(W, sparse.csr_matrix):
            W.data[W.data < threshold] = 0
            W.eliminate_zeros()
        else:
            W[W < threshold] = 0

        cids = utils.get_connected_component(W)
        author_paper_table = author_paper_table.copy()
        author_paper_table["cluster_id"] = [
            "%s_%d" % (block_name, c + maxcid) for c in cids
        ]
        res_table_list += [author_paper_table]
        maxcid += len(cids)

    #
    # Save
    #
    res_table = pd.concat(res_table_list, ignore_index=True)
    res_table.to_csv(OUTPUT, index=False)
