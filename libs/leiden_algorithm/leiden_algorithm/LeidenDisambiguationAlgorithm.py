import numpy as np
import networkx as nx
from scipy import sparse
import pandas as pd

import glob
import shutil

import os
import sys
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from .utils import *
from .DataBlockingAlgorithm import DataBlockingAlgorithm
from .ScoringRule import ScoringRule


class LeidenDisambiguationAlgorithm:
    """
    Leiden Disambiguation Algorithm. 

    Example
    -------

    lda = LeidenDisambiguationAlgorithm(
        "working_directory", ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB, general_name_list
    )

    lda.init_working_dir()
    lda.data_blocking(wos_ids)
    lda.clustering()
    lda.post_process()
    
    disambiguted_author_list = lda.get_disambiguated_authors()
    """

    def __init__(
        self,
        working_dir,
        ES_USERNAME,
        ES_PASSWORD,
        ES_ENDPOINT,
        CITATION_DB,
        general_name_list,
        threshold=10,
    ):
        """
        Params
        ------
        working_dir: str
            Name of working directory that contains
                - blocks : this contains sql databases for data blocks
                - clustered : this contains the csv files for clustered sub-blocks
                - disambiguated: this contains the final distambiguated result
        ES_USERNAE: str
            Username for the ElasticSearch server
        ES_PASSWORD: str
            Password for the ElasticSearch server
        CITATION_DB: str
            Path to the sql database for citations
        general_name_list: str
            List of general names
        threshold: float
            Threshold value for clustering
        """
        self.data_blocking_alg = DataBlockingAlgorithm(
            ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB
        )
        self.working_dir = working_dir
        self.general_name_list = general_name_list
        self.blocks_dir = "%s/blocks" % self.working_dir
        self.clustered_dir = "%s/clustered_blocks" % self.working_dir
        self.disambiguated_dir = "%s/disambiguated_blocks" % self.working_dir
        self.disambiguated_list = (
            self.disambiguated_dir + "/disambiguated-author-list.csv"
        )
        self.threshold = threshold

    def init_working_dir(self):

        try:
            shutil.rmtree(self.working_dir)
        except OSError as e:
            pass
            # print("Error: %s : %s" % (self.working_dir, e.strerror))

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        os.makedirs(self.blocks_dir)
        os.makedirs(self.clustered_dir)
        os.makedirs(self.disambiguated_dir)

    def data_blocking(self, wos_ids, max_chunk=1000):
        num_partitions = np.ceil(len(wos_ids) / max_chunk).astype(int)
        for pid in tqdm(range(num_partitions), desc="Grouping data"):
            n0 = pid * max_chunk
            n1 = np.minimum(n0 + max_chunk, len(wos_ids))
            sub_wos_ids = wos_ids[n0:n1]
            self.data_blocking_alg.run(sub_wos_ids, self.blocks_dir)

    def clustering(self, block_list=[]):
        def clustering_method(self, W):
            W[W < self.threshold] = 0
            return get_connected_component(W)

        if len(block_list) == 0:
            block_list = glob.glob(self.blocks_dir + "/*.db")
            block_list = [Path(b).stem for b in block_list]

        for block_name in tqdm(block_list, desc="Clustering"):
            dbname = self.blocks_dir + "/" + block_name + ".db"
            result_file_name = self.clustered_dir + "/" + block_name + ".csv"

            scoring_func = ScoringRule(self.general_name_list)
            W, author_paper_table = scoring_func.eval(
                self.blocks_dir + "/" + block_name + ".db"
            )

            cids = get_connected_component(W)

            author_paper_table = author_paper_table.copy()
            author_paper_table["cluster_id"] = ["%s_%d" % (block_name, c) for c in cids]
            author_paper_table.to_csv(result_file_name, index=False)

    def post_process(self, block_list=[], n_jobs=10):
        if len(block_list) == 0:
            block_list = glob.glob(self.clustered_dir + "/*.csv")
            block_list = [Path(b).stem for b in block_list]

        df = (
            pd.concat(
                Parallel(n_jobs)(
                    delayed(pd.read_csv)(self.clustered_dir + "/%s.csv" % b)
                    for b in block_list
                )
            )
            .reset_index()
            .drop(columns=["index"])
        )

        # Convert cluster_id to integer id
        df = df.rename(columns={"cluster_id": "tmp_id"})
        cluster_ids = df[["tmp_id"]].drop_duplicates()
        cluster_num = cluster_ids.size
        cluster_ids["cluster_id"] = np.arange(cluster_num)
        df = pd.merge(df, cluster_ids, on="tmp_id", how="left")
        df = df.drop(columns=["tmp_id"])

        # Clustering by a co-occurrence network of clusters with email_address
        W = to_cooccurrence_matrix(
            df, "cluster_id", "email_address", num_rows=cluster_num
        )
        W.setdiag(1)
        cids = get_connected_component(W)
        df = pd.merge(
            df,
            pd.DataFrame(
                {"cluster_id": np.arange(cids.size), "disambiguated_author_id": cids}
            ),
            on="cluster_id",
            how="left",
        )
        df = df[["paper_id", "name", "disambiguated_author_id"]]
        df.to_csv(
            self.disambiguated_dir + "/disambiguated-author-list.csv",
            sep=",",
            index=False,
        )

    def get_disambiguated_authors(self):

        if os.path.exists(self.disambiguated_list):
            return pd.read_csv(self.disambiguated_list)
        return None
