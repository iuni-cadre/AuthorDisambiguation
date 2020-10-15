import glob
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join("libs/leiden_algorithm")))
from leiden_algorithm import LeidenDisambiguationAlgorithm


if __name__ == "__main__":

    INPUT_JSON_DIR = sys.argv[1]
    CITATION_DB = sys.argv[2]
    GENERAL_NAME_LIST_FILE = sys.argv[3]
    WORKING_DIR = sys.argv[4]
    OUTPUT = sys.argv[5]

    json_files = glob.glob(INPUT_JSON_DIR+"/*.json")

    general_name_list = pd.read_csv(GENERAL_NAME_LIST_FILE)["first_name"].values

    lda = LeidenDisambiguationAlgorithm(
        WORKING_DIR,
        CITATION_DB,
        general_name_list,
        n_jobs=50,
    )

    lda.init_working_dir()
    lda.data_blocking(json_files)
    lda.clustering()
    lda.post_process()

    disambiguted_author_list = lda.get_disambiguated_authors()
    disambiguted_author_list.to_csv(OUTPUT, index=False)
