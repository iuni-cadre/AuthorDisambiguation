import numpy as np
import pandas as pd
import utils
from scipy import sparse
from ScoringRule import ScoringRule

if __name__ == "__main__":

    GENERAL_NAME_LIST_FILE = snakemake.input["general_name_list"]
    root = snakemake.input["data_path"]
    block_name = snakemake.params["block_name"]
    threshold = float(snakemake.params["threshold"])
    OUTPUT = snakemake.output["output_file"]

    #
    # Load
    #
    general_name_list = pd.read_csv(GENERAL_NAME_LIST_FILE)["first_name"].values

    #
    # Clustering
    #
    # lda.clustering(block_list = ["initials2=ada"], root = dirpath)

    scoring_func = ScoringRuleCSV(general_name_list, root)
    W, author_paper_table = scoring_func.eval(block_name)

    W[W < threshold] = 0
    cids = utils.get_connected_component(W)

    #
    # Save
    #
    author_paper_table = author_paper_table.copy()
    author_paper_table["cluster_id"] = ["%s_%d" % (block_name, c) for c in cids]
    author_paper_table.to_csv(OUTPUT, index=False)
