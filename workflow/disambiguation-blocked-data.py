
import glob
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join("libs/leiden_algorithm")))
from leiden_algorithm import LeidenDisambiguationAlgorithm


if __name__ == "__main__":

    GENERAL_NAME_LIST_FILE = snakemake.input["general_name_list"]
    WORKING_DIR = snakemake.output["working_dir"]
    OUTPUT = snakemake.output["disambiguated_author_list"]

    general_name_list = pd.read_csv(GENERAL_NAME_LIST_FILE)["first_name"].values

    lda = LeidenDisambiguationAlgorithmCSV(
        WORKING_DIR,
        None,#CITATION_DB,
        general_name_list,
        n_jobs=50,
    )

    lda.init_working_dir()
    #lda.data_blocking(json_files)
    lda.clustering(block_list = ["initals2=ada"])
    lda.post_process()

    disambiguted_author_list = lda.get_disambiguated_authors()
    disambiguted_author_list.to_csv(OUTPUT, index=False)
