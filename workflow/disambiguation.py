import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join("libs/leiden_algorithm")))
from leiden_algorithm import LeidenDisambiguationAlgorithm


if __name__ == "__main__":

    ES_USERNAME = sys.argv[1]
    ES_PASSWORD = sys.argv[2]
    ES_ENDPOINT = sys.argv[3]
    INPUT_WOS_FILE = sys.argv[4]
    WOS_ID_COLUMN_NAME = sys.argv[5] 
    CITATION_DB = sys.argv[6]
    GENERAL_NAME_LIST_FILE = sys.argv[7]
    WORKING_DIR = sys.argv[8]
    OUTPUT = sys.argv[9]

    wos_ids = pd.read_csv(INPUT_WOS_FILE)[WOS_ID_COLUMN_NAME].drop_duplicates().values.tolist()

    general_name_list = pd.read_csv(GENERAL_NAME_LIST_FILE)["first_name"].values

    lda = LeidenDisambiguationAlgorithm(
        WORKING_DIR,
        ES_USERNAME,
        ES_PASSWORD,
        ES_ENDPOINT,
        CITATION_DB,
        general_name_list,
    )

    lda.init_working_dir()
    lda.data_blocking(wos_ids)
    lda.clustering()
    lda.post_process()

    disambiguted_author_list = lda.get_disambiguated_authors()
    disambiguted_author_list.to_csv(OUTPUT, index=False)
