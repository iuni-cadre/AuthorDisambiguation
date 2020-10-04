import numpy as np
from scipy import sparse
import pandas as pd
from DataBlockingAlgorithm import DataBlockingAlgorithm

# from scoring import DataBase, ScoringRule
import shutil
import os
import sys
from tqdm import tqdm


class LeidenDisambiguationAlgorithm:
    def __init__(
        self,
        working_dir,
        ES_USERNAME,
        ES_PASSWORD,
        ES_ENDPOINT,
        CITATION_DB,
        db,
        general_name_list,
    ):
        self.data_blocking_alg = DataBlockingAlgorithm(
            ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB
        )
        self.working_dir = working_dir
        # self.scoring_func = self.ScoringRule(db, general_name_list)
        # self.general_name_list
        self.blocks_dir = "%s/blocks" % self.working_dir
        self.clustered_dir = "%s/clustered_blocks" % self.working_dir
        self.disambiguated_dir = "%s/disambiguated_blocks" % self.working_dir

    def init_working_dir(self):

        try:
            shutil.rmtree(self.working_dir)
        except OSError as e:
            print("Error: %s : %s" % (self.working_dir, e.strerror))

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        os.makedirs(self.blocks_dir)
        os.makedirs(self.clustered_dir)
        os.makedirs(self.disambiguated_dir)

    def data_blocking(self, wos_ids, max_chunk=10):
        num_partitions = np.ceil(len(wos_ids) / max_chunk).astype(int)
        for pid in tqdm(range(num_partitions), desc="Grouping data"):
            n0 = pid * max_chunk
            n1 = np.minimum(n0 + max_chunk, len(wos_ids))
            sub_wos_ids = wos_ids[n0:n1]
            self.data_blocking_alg.run(sub_wos_ids, self.blocks_dir)


if __name__ == "__main__":

    WOS_ID_FILE = "../data/testData.csv"
    CITATION_DB = "../data/wos-citation.db"
    ES_PASSWORD = "FSailing4046"
    ES_USERNAME = "skojaku"
    ES_ENDPOINT = "localhost:9200/wos/_search/"

    # Retrieve the wos_ids
    wos_ids = pd.read_csv(WOS_ID_FILE)["UID"].drop_duplicates().values.tolist()

    lda = LeidenDisambiguationAlgorithm(
        "tmp", ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB, [], []
    )

    lda.init_working_dir()

    lda.data_blocking(wos_ids)
