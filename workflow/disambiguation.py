import numpy as np
import yaml
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join("libs/leiden_algorithm")))
from leiden_algorithm import LeidenDisambiguationAlgorithm


if __name__ == "__main__":

    ES_CONFIG_FILE = sys.argv[1]
    INPUT_WOS_FILE = sys.argv[2]
    WOS_ID_COLUMN_NAME = sys.argv[3] 
    CITATION_DB = sys.argv[4]
    GENERAL_NAME_LIST_FILE = sys.argv[5]
    WORKING_DIR = sys.argv[6]
    OUTPUT = sys.argv[7]

    with open(ES_CONFIG_FILE) as f:
        ES = yaml.safe_load(f)
        ES_USERNAME = ES["es_username"] 
        ES_PASSWORD = ES["es_password"] 
        ES_ENDPOINT = ES["es_endpoint"] 

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
