import numpy as np
from scipy import sparse
import pandas as pd
import sqlite3


#
# SQL interface
#

# a decolator for Interface class methods
def impute_nan_to_sql_table(func):
    """replace 'nan' string with python nan """

    def wrapper(*args, **kwargs):
        tb = func(*args, **kwargs).replace("nan", np.nan)
        return tb

    return wrapper


class DataBase:
    def __init__(self, conn):
        self.conn = conn

    def get_author_paper_list_in_block(self, block_id):
        # comments : can I perform these operations with only the sql?
        # But doing this may require more memory because the join clause will be pefermed after where clause
        name_table = pd.read_sql(
            "select * from name_table where block_id = {block_id}".format(
                block_id=block_id
            ),
            self.conn,
        )
        name_id_table = pd.read_sql(
            "select * from name_paper_table where block_id = {block_id}".format(
                block_id=block_id
            ),
            self.conn,
        ).drop(columns=["block_id", "short_name_id"])
        return pd.merge(name_table, name_id_table, on="name_id", how="right")

    def get_name_ids_in_block(self, block_id):
        name_table = pd.read_sql(
            "select * from name_table where block_id = {block_id}".format(
                block_id=block_id
            ),
            self.conn,
        )
        return name_table["name_id"].values

    @impute_nan_to_sql_table
    def get_paper_attributes(self, paper_ids):
        return pd.read_sql(
            "select * from paper_table where paper_id in ({paper_ids})".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )

    @impute_nan_to_sql_table
    def get_grants(self, paper_ids):
        return pd.read_sql(
            "select * from grant_table where paper_id in ({paper_ids})".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )

    @impute_nan_to_sql_table
    def get_citations(self, paper_ids):
        citations = pd.read_sql(
            "select * from citation_table where source in ({paper_ids}) or target in ({paper_ids})".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )
        return citations.dropna()

    @impute_nan_to_sql_table
    def get_authors(self, paper_ids):
        # comments : can I perform these operations with only the sql?
        # But doing this may require more memory because the join clause will be pefermed after where clause
        return pd.read_sql(
            "select name_id, short_name_id, paper_id from name_paper_table where paper_id in ({paper_ids})".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )

    @impute_nan_to_sql_table
    def get_paper_address(self, paper_ids):
        # comments : can I perform these operations with only the sql?
        # But doing this may require more memory because the join clause will be pefermed after where clause
        return pd.read_sql(
            "select * from paper_address_table where paper_id in ({paper_ids}) ".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )

    @impute_nan_to_sql_table
    def get_name_paper_address(self, paper_ids):
        # comments : can I perform these operations with only the sql?
        # But doing this may require more memory because the join clause will be pefermed after where clause
        return pd.read_sql(
            "select * from name_paper_address_table where paper_id in ({paper_ids}) ".format(
                paper_ids=to_string_array(paper_ids),
            ),
            self.conn,
        )


#
# Scoring functions
#
# tables = {author_paper_table, name_paper_address, paper_address, paper_table, grant_table, coauthor_table}
#


class ScoringRule:
    def __init__(self, db, general_name_list):
        self.db = db
        self.general_name_list = general_name_list

        self.author_paper_table = None
        self.paper_address_table = None
        self.name_paper_address_table = None
        self.paper_table = None
        self.grant_table = None
        self.coauthor_table = None
        self.citation_table = None

    def load_tables(self, block_id):
        # Get (name, paper) pairs

        self.author_paper_table = self.db.get_author_paper_list_in_block(block_id)
        paper_ids = self.author_paper_table["paper_id"].drop_duplicates().values

        # Assign index values
        self.author_paper_table["index"] = np.arange(self.author_paper_table.shape[0])

        # Load tables
        self.paper_address_table = self.db.get_paper_address(paper_ids)
        self.name_paper_address_table = self.db.get_name_paper_address(paper_ids)
        self.paper_table = self.db.get_paper_attributes(paper_ids)
        self.grant_table = self.db.get_grants(paper_ids)
        self.coauthor_table = self.db.get_authors(paper_ids)
        self.citation_table = self.db.get_citations(paper_ids)

    def eval(self, block_id):

        self.load_tables(block_id)
        rule_list = [
            self.rule_1,
            self.rule_2,
            self.rule_3,
            self.rule_4,
            self.rule_5,
            self.rule_6,
            self.rule_7,
            self.rule_8,
            self.rule_9_10_11,
        ]

        W = None
        for i, rule in enumerate(rule_list):
            Wnew = rule_list[i]()
            if Wnew is None:
                continue
            elif W is None:
                W = Wnew
            else:
                W += Wnew
        return W, self.author_paper_table[["name_id", "paper_id", "name_paper_id"]]

    def rule_1(self):
        return to_weighted_cooccurrence_matrix(
            self.author_paper_table,
            "index",
            ["email_address"],
            [100],
            num_rows=self.author_paper_table.shape[0],
        )

    def rule_2(self):
        def generate_initials_more_than_two(x):
            first_name = x["first_name"] if x["first_name"] is not None else ""
            last_name = x["last_name"] if x["last_name"] is not None else ""
            s = "".join([s[0] for s in (first_name + " " + last_name).split()])
            if len(s) > 2:
                return s
            return None

        self.author_paper_table[
            "initials_more_than_two"
        ] = self.author_paper_table.apply(
            lambda x: generate_initials_more_than_two(x), axis=1
        )

        # Rule 2c
        Wn = to_weighted_cooccurrence_matrix(
            self.author_paper_table,
            "index",
            ["initials"],
            [1],
            num_rows=self.author_paper_table.shape[0],
        )
        return to_nested_weighted_cooccurrence_matrix(
            self.author_paper_table,
            "index",
            ["initials", "initials_more_than_two"],
            [5, 10],
            num_rows=self.author_paper_table.shape[0],
        ) - 10 * (1 - Wn)

    def rule_3(self):
        self.author_paper_table[
            "is_general_first_name"
        ] = self.author_paper_table.apply(
            lambda x: x["first_name"]
            if x["first_name"] in self.general_name_list
            else None,
            axis=1,
        )
        self.author_paper_table[
            "is_special_first_name"
        ] = self.author_paper_table.apply(
            lambda x: x["first_name"] if x["is_general_first_name"] is None else None,
            axis=1,
        )
        return to_nested_weighted_cooccurrence_matrix(
            self.author_paper_table,
            "index",
            ["is_general_first_name", "is_special_first_name"],
            [3, 6],
            num_rows=self.author_paper_table.shape[0],
        )

    def rule_4(self):
        address_table = pd.merge(
            self.author_paper_table.reset_index(),
            self.name_paper_address_table,
            on="name_paper_id",
        )
        if address_table.shape[0] > 0:

            # Merge fields
            address_fields = ["country", "city", "organization", "department"]

            def get_address_name(x, field_names):
                retval = []
                for field in field_names:
                    if pd.isna(x[field]):
                        return None
                    retval += [x[field]]
                return "_".join(retval)

            for i, c in enumerate(address_fields):
                if i > 0:
                    address_table[
                        "_".join(address_fields[: i + 1])
                    ] = address_table.apply(
                        lambda x: get_address_name(x, address_fields[: i + 1]), axis=1
                    )
            return to_nested_weighted_cooccurrence_matrix(
                address_table,
                "index",
                [
                    "country_city",
                    "country_city_organization",
                    "country_city_organization_department",
                ],
                [4, 7, 10],
                num_rows=self.author_paper_table.shape[0],
            )
        return None

    def rule_5(self):
        #
        # Rule 5a, b, c
        #
        df = pd.concat(
            [
                self.coauthor_table[["paper_id", "short_name_id"]],
                self.author_paper_table[["paper_id", "short_name_id"]],
            ]
        )
        df = pd.merge(
            df,
            self.author_paper_table[["index", "paper_id"]],
            on="paper_id",
            how="left",
        )

        paper_author_mat, pids, _ = to_binary_matrix(
            df["index"].values, df["short_name_id"].values
        )
        paper_author_mat.data[
            paper_author_mat.data == 2
        ] = 0  # to exclude the authors to be disambiguated

        coauthor_count_paper_by_paper = paper_author_mat @ paper_author_mat.T
        coauthor_count_paper_by_paper.data[coauthor_count_paper_by_paper.data > 2] = 10
        coauthor_count_paper_by_paper.data[coauthor_count_paper_by_paper.data == 2] = 7
        coauthor_count_paper_by_paper.data[coauthor_count_paper_by_paper.data == 1] = 4
        return coauthor_count_paper_by_paper.toarray()

    def rule_6(self):
        df = pd.merge(
            self.author_paper_table, self.grant_table, on="paper_id", how="left"
        )
        return (
            10
            * to_cooccurrence_matrix(
                df, "index", "grant_id", num_rows=self.author_paper_table.shape[0]
            ).toarray()
        )

    def rule_7(self):

        # Rules 7a, 7b
        address_table = pd.merge(
            self.author_paper_table, self.paper_address_table, on="paper_id"
        )
        if address_table.shape[0] > 0:
            # Merge fields
            address_fields = ["country", "city", "organization", "department"]

            def get_address_name(x, field_names):
                retval = []
                for field in field_names:
                    if pd.isna(x[field]):
                        return None
                    retval += [x[field]]
                return "_".join(retval)

            for i, c in enumerate(address_fields):
                if i > 0:
                    address_table[
                        "_".join(address_fields[: i + 1])
                    ] = address_table.apply(
                        lambda x: get_address_name(x, address_fields[: i + 1]), axis=1
                    )
            W = to_nested_weighted_cooccurrence_matrix(
                address_table,
                "index",
                [
                    "country_city",
                    "country_city_organization",
                    "country_city_organization_department",
                ],
                [4, 7, 10],
                num_rows=author_paper_table.shape[0],
            )
            return W
        else:
            return None

    def rule_8(self):
        df = pd.merge(
            self.author_paper_table,
            self.paper_table[["paper_id", "journal"]],
            on="paper_id",
            how="left",
        )
        df["index"] = np.arange(df.shape[0])
        W = to_weighted_cooccurrence_matrix(
            df, "index", ["journal"], [6], num_rows=df.shape[0]
        )
        return W

    def rule_9_10_11(self):

        Wlist = []
        #
        # Rules 10a, b, c, d, e
        #
        cited_paper_table = pd.merge(
            self.author_paper_table[["paper_id", "index"]].rename(
                columns={"paper_id": "target"}
            ),
            self.citation_table,
            on="target",
            how="left",
        ).dropna()
        bib_coupling_count_mat = 2 * to_cooccurrence_matrix(
            cited_paper_table,
            "index",
            "source",
            num_rows=self.author_paper_table.shape[0],
            binarize=False,
        )
        bib_coupling_count_mat.data = np.minimum(bib_coupling_count_mat.data, 10)
        Wlist += [bib_coupling_count_mat]

        #
        # Rules 11a, b, c, d, e
        #
        citing_paper_table = pd.merge(
            self.author_paper_table[["paper_id", "index"]].rename(
                columns={"paper_id": "source"}
            ),
            self.citation_table,
            on="source",
            how="left",
        ).dropna()
        co_citation_count_mat = to_cooccurrence_matrix(
            citing_paper_table,
            "index",
            "target",
            num_rows=self.author_paper_table.shape[0],
            binarize=False,
        )
        co_citation_count_mat.data = np.minimum(co_citation_count_mat.data + 1, 6)
        Wlist += [co_citation_count_mat]

        #
        # Rules 9
        #
        citing_paper_table = citing_paper_table.iloc[
            np.isin(
                citing_paper_table["target"].values,
                self.author_paper_table["paper_id"].values,
            )
        ]
        W = 10 * to_cooccurrence_matrix(
            citing_paper_table,
            "index",
            "target",
            num_rows=self.author_paper_table.shape[0],
            binarize=True,
        )
        Wlist += [W]
        W = np.sum(Wlist, axis=0).toarray()
        return W


#
# Helper functions
#
def to_binary_matrix(source, target, num_rows=None, num_cols=None):

    non_missing = (~pd.isna(source)) * (~pd.isna(target))
    source = source[non_missing]
    target = target[non_missing]

    if len(source) == 0:
        if num_rows is None:
            num_rows = 1
        if num_cols is None:
            num_cols = 1
        shape = (num_rows, num_cols)
        return (
            sparse.csr_matrix(([0], ([0], [0])), shape=shape,),
            [],
            [],
        )

    if isinstance(source[0], int) is False:
        source_keys, source_id = np.unique(source, return_inverse=True)
    else:
        source_keys = None
        source_id = source

    if isinstance(target[0], int) is False:
        target_keys, target_id = np.unique(target, return_inverse=True)
    else:
        target_keys = None
        target_id = target

    if num_rows is None:
        num_rows = np.max(source_id) + 1
    if num_cols is None:
        num_cols = np.max(target_id) + 1

    shape = (num_rows, num_cols)

    return (
        sparse.csr_matrix(
            (np.ones_like(source_id), (source_id, target_id)), shape=shape,
        ),
        source_keys,
        target_keys,
    )


def to_cooccurrence_matrix(
    df, source, target, num_rows=None, num_cols=None, binarize=True
):
    B, _, _ = to_binary_matrix(
        df[source].values, df[target].values, num_rows=num_rows, num_cols=num_cols
    )
    C = B @ B.T
    if binarize:
        C.data = np.ones_like(C.data)
    return C


def to_weighted_cooccurrence_matrix(
    df, id_col, columns, weights, num_rows=None, return_mat="dense"
):
    """
    Calculate the co-occurence of items 
    
    Params
    ------
    df: pandas.DataFrame
    id_col: str
        Column for the index of the co-occurrence matrix
    columns: list of str
        Name of columns used for computing the matching score
    weights: list of float 
        The matching score is given by 
            sum_{c in columns} w[c] * delta(df.loc[i, c], df.loc[j, c])
        where delta is the Kronecker delta
    num_cols: number of columns
    num_rows: number of rows
        
    Return
    ------
    W : numpy.array or scipy.sparse.csr_matrix
        (N by N) matrix, where W[i,j] is the matching score for pairs (df.iloc[i], df.iloc[j])
    """
    W = None
    for i, w in enumerate(weights):
        if i == 0:
            W = w * to_cooccurrence_matrix(df, id_col, columns[i], num_rows=num_rows)
        else:
            Wnew = to_cooccurrence_matrix(df, id_col, columns[i], num_rows=num_rows)
            W += w * Wnew
    if return_mat == "dense":
        return W.toarray()
    elif return_mat == "sparse":
        return W


def to_nested_weighted_cooccurrence_matrix(df, id_col, columns, weights, num_rows=None):
    """
    Calculate the co-occurence of items. 
    The weights for matching rules have a hierarchy, where columns[i+1] is the upper group of columns[i].
    When a pair satisfies more than two rules simultaneously, the weight of the upper most rules will be used.
    """
    if isinstance(weights, list):
        weights = np.array(weights)
    W = to_weighted_cooccurrence_matrix(
        df,
        id_col,
        columns,
        np.power(10, np.arange(len(columns))),
        num_rows=num_rows,
        return_mat="sparse",
    )
    if W.count_nonzero() == 0:
        return W.toarray()

    # Find the length of each element
    length_checker = np.vectorize(len)
    W.data = length_checker(W.data.astype(int).astype(str))
    W.data = weights[W.data.astype(int) - 1]
    return W.toarray()


def to_string_array(a):
    """from numpy array to string compatible with sql in c"""
    return ",".join(['"%s"' % b for b in a])
