import numpy as np
from scipy import sparse
import pandas as pd
import sqlite3


#
# To retrieve paper-name pairs in blocks
#
def get_author_paper_list_in_block(block_id, conn):
    # comments : can I perform these operations with only the sql?
    # But doing this may require more memory because the join clause will be pefermed after where clause
    name_table = pd.read_sql(
        "select * from name_table where block_id = {block_id}".format(
            block_id=block_id
        ),
        conn,
    )
    name_id_table = pd.read_sql(
        "select * from name_paper_table where block_id = {block_id}".format(
            block_id=block_id
        ),
        conn,
    ).drop(columns=["block_id", "short_name_id"])
    return pd.merge(name_table, name_id_table, on="name_id", how="right")


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


def to_cooccurrence_matrix(source, target, num_rows=None, num_cols=None, binarize=True):
    B, _, _ = to_binary_matrix(source, target, num_rows=num_rows, num_cols=num_cols)
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
            W = w * to_cooccurrence_matrix(
                df[id_col].values, df[columns[i]].values, num_rows=num_rows
            )
        else:
            Wnew = to_cooccurrence_matrix(
                df[id_col].values, df[columns[i]].values, num_rows=num_rows
            )
            W += w * to_cooccurrence_matrix(
                df[id_col].values, df[columns[i]].values, num_rows=num_rows
            )
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


#
# SQL interface
#
def get_name_ids_in_block(block_id, conn):
    name_table = pd.read_sql(
        "select * from name_table where block_id = {block_id}".format(
            block_id=block_id
        ),
        conn,
    )
    return name_table["name_id"].values


def impute_nan_to_sql_table(func):
    """replace 'nan' string with python nan """

    def wrapper(*args, **kwargs):
        tb = func(*args, **kwargs).replace("nan", np.nan)
        return tb

    return wrapper


@impute_nan_to_sql_table
def get_paper_attributes(paper_ids, conn):
    return pd.read_sql(
        "select * from paper_table where paper_id in ({paper_ids})".format(
            paper_ids=to_string_array(paper_ids),
        ),
        conn,
    )


@impute_nan_to_sql_table
def get_citations(paper_ids, conn):
    citations = pd.read_sql(
        "select * from citation_table where source in ({paper_ids}) or target in ({paper_ids})".format(
            paper_ids=to_string_array(paper_ids),
        ),
        conn,
    )
    return citations.dropna()


@impute_nan_to_sql_table
def get_authors(paper_ids, conn):
    # comments : can I perform these operations with only the sql?
    # But doing this may require more memory because the join clause will be pefermed after where clause
    return pd.read_sql(
        "select name_id, short_name_id, paper_id from name_paper_table where paper_id in ({paper_ids})".format(
            paper_ids=to_string_array(paper_ids),
        ),
        conn,
    )


@impute_nan_to_sql_table
def get_paper_address(paper_ids, conn):
    # comments : can I perform these operations with only the sql?
    # But doing this may require more memory because the join clause will be pefermed after where clause
    return pd.read_sql(
        "select * from paper_address_table where paper_id in ({paper_ids}) ".format(
            paper_ids=to_string_array(paper_ids),
        ),
        conn,
    )


@impute_nan_to_sql_table
def get_name_paper_address(paper_ids, conn):
    # comments : can I perform these operations with only the sql?
    # But doing this may require more memory because the join clause will be pefermed after where clause
    return pd.read_sql(
        "select * from name_paper_address_table where paper_id in ({paper_ids}) ".format(
            paper_ids=to_string_array(paper_ids),
        ),
        conn,
    )


#
# Paiwise matching functions
#
def pairwise_matching(df, column):
    vals = df[column].values
    non_missing = np.where(~pd.isna(vals))[0]
    _, val_ids = np.unique(vals[non_missing], return_inverse=True)
    num_keys = len(non_missing)
    num_uniq_vals = len(val_ids)
    U = sparse.csr_matrix(
        (np.ones(num_keys), (non_missing, val_ids)), shape=(df.shape[0], num_uniq_vals),
    )
    # Matching
    W = U @ U.T
    return W


def weighted_matching(df, columns, weights, return_mat="dense"):
    """
    Calculate the matching score for pairs of rows 
    
    Params
    ------
    df: pandas.DataFrame
    columns: list of str
        Name of columns used for computing the matching score
    weights: list of float 
        The matching score is given by 
            sum_{c in columns} w[c] * delta(df.loc[i, c], df.loc[j, c])
        where delta is the Kronecker delta
    return_mat: 'dense' or 'sparse'
        return_mat='dense' and ='sparse' returns a numpy array and scipy.sparse.csr_matrix, respectively
        
    Return
    ------
    W : numpy.array or scipy.sparse.csr_matrix
        (N by N) matrix, where W[i,j] is the matching score for pairs (df.iloc[i], df.iloc[j])
    """
    W = None
    for i, w in enumerate(weights):
        if i == 0:
            W = w * pairwise_matching(df, columns[i])
        else:
            W += w * pairwise_matching(df, columns[i])
    if return_mat == "dense":
        return W.toarray()
    elif return_mat == "sparse":
        return W


def nested_weighted_matching(df, columns, weights):
    """
    Calculate the matching score for pairs of rows. 
    The weights for matching rules have a hierarchy, where columns[i+1] is the upper group of columns[i].
    When a pair satisfies more than two rules simultaneously, the weight of the upper most rules will be used.
    """
    if isinstance(weights, list):
        weights = np.array(weights)
    W = weighted_matching(
        df, columns, np.power(10, np.arange(len(columns))), return_mat="sparse"
    )
    if W.count_nonzero() == 0:
        return W.toarray()

    # Find the length of each element
    length_checker = np.vectorize(len)
    W.data = length_checker(W.data.astype(int).astype(str))
    W.data = weights[W.data.astype(int) - 1]
    return W.toarray()


#
# Scoring functions
#
def matching_score_by_authors(author_paper_table, general_name_list, conn):
    """
    Scoreing rules for authors.
    Implemented rules: 1, 2a, 2b, 2c, 3a, 3b
    Not implemented rules: 4a, 4b, 4c
    """

    def generate_initials_more_than_two(x):
        first_name = x["first_name"] if x["first_name"] is not None else ""
        last_name = x["last_name"] if x["last_name"] is not None else ""
        s = "".join([s[0] for s in (first_name + " " + last_name).split()])
        if len(s) > 2:
            return s
        return None

    #
    # Get paper ids
    #
    paper_ids = author_paper_table["paper_id"].drop_duplicates().values
    num_papers = len(paper_ids)
    address_table = get_name_paper_address(paper_ids, conn)

    author_attri = author_paper_table.copy()
    author_attri["initials_more_than_two"] = author_attri.apply(
        lambda x: generate_initials_more_than_two(x), axis=1
    )
    author_attri["is_general_first_name"] = author_attri.apply(
        lambda x: x["first_name"] if x["first_name"] in general_name_list else None,
        axis=1,
    )
    author_attri["is_special_first_name"] = author_attri.apply(
        lambda x: x["first_name"] if x["is_general_first_name"] is None else None,
        axis=1,
    )

    Wlist = []  # List of score matrices

    # Rule 1
    W = weighted_matching(author_attri, ["email_address"], [100])
    Wlist += [W]

    # Rule 2a and 2b
    Wlist += [
        nested_weighted_matching(
            author_attri, ["initials", "initials_more_than_two"], [5, 10]
        )
    ]

    # Rule 2c
    Wn = weighted_matching(author_attri, ["initials"], [1])
    Wlist += [-10 * (1 - Wn)]

    # Rule 3a and 3b
    W = nested_weighted_matching(
        author_attri, ["is_general_first_name", "is_special_first_name"], [3, 6]
    )
    Wlist += [W]

    # Rule 4a and 4b
    # (not implemented yet)
    address_table = pd.merge(
        author_attri.reset_index(), address_table, on="name_paper_id"
    )

    # Merge fields
    address_fields = ["country", "city", "organization", "department"]
    # address_table = address_table.replace('nan', np.nan)
    def get_address_name(x, field_names):
        retval = []
        for field in field_names:
            if pd.isna(x[field]):
                return None
            retval += [x[field]]
        return "_".join(retval)

    for i, c in enumerate(address_fields):
        if i > 0:
            address_table["_".join(address_fields[: i + 1])] = address_table.apply(
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
        num_rows=author_attri.shape[0],
    )
    Wlist += [W]

    # Sum
    W = np.sum(Wlist, axis=0)

    return W


def matching_score_by_articles(author_paper_table, block_id, conn):
    """
    Scoring based on the rules for articles. 
    Implemented rules:  5a, 5b, 5c, 7a, 8a, 8b
    Not implemented rules: 6, 7b, 7c 
    """
    # Implementation note
    # -------------------
    # The rules compute the scores for pairs of papers. Therefore, we first compute the
    # scores for pairs of papers. Then, distribute the scores at (name-paper)x(name-paper) level.

    #
    # Get paper ids
    #
    paper_ids = author_paper_table["paper_id"].drop_duplicates().values
    num_papers = len(paper_ids)

    #
    # Get tables
    #
    address_table = get_paper_address(paper_ids, conn)
    coauthor_table = get_authors(paper_ids, conn)

    Wlist = []
    #
    # Rule 5a, b, c
    #
    df = pd.concat(
        [
            coauthor_table[["paper_id", "short_name_id"]],
            author_paper_table[["paper_id", "short_name_id"]],
        ]
    )
    df = pd.merge(
        df,
        author_paper_table.reset_index()[["index", "paper_id"]],
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
    Wlist += [coauthor_count_paper_by_paper]

    #
    # Rule 6
    #
    # (not implemented yet)

    #
    # Rule 7a
    #
    address_table["country_city"] = (
        address_table["country"] + "_" + address_table["city"]
    )

    # a binary matrix of papers and address, where 1 indicates that the address is associated with the paper
    paper_address_mat, pids, _ = to_binary_matrix(
        address_table["paper_id"].values,
        address_table["country_city"].values,
        num_rows=num_papers,
    )

    # Count the address
    address_count_paper_by_paper = paper_address_mat @ paper_address_mat.T
    address_count_paper_by_paper.data = (
        np.ones_like(address_count_paper_by_paper.data) * 2
    )

    # distribute the score for papers to name-paper pairs
    name_paper_by_paper, paper_ids, _ = to_binary_matrix(
        author_paper_table["paper_id"].values,
        np.arange(author_paper_table.shape[0]),
        num_rows=num_papers,
    )
    address_count_paper_by_paper = (
        name_paper_by_paper.T @ address_count_paper_by_paper @ name_paper_by_paper
    )

    Wlist += [address_count_paper_by_paper]

    # Rules 7a, 7b
    # (not implemented yet)

    # Sum
    W = np.sum(Wlist, axis=0)
    return W


def matching_score_by_source(author_paper_table, conn):
    paper_ids = author_paper_table["paper_id"].drop_duplicates().values
    paper_table = get_paper_attributes(paper_ids, conn)
    df = pd.merge(
        author_paper_table,
        paper_table[["paper_id", "journal"]],
        on="paper_id",
        how="left",
    )
    W = weighted_matching(df, ["journal"], [6])
    return W


def matching_score_by_citations(author_paper_table, conn):

    #
    # Get paper ids
    #
    paper_ids = author_paper_table["paper_id"].drop_duplicates().values
    num_papers = len(paper_ids)

    #
    # Get table
    #
    citation_table = get_citations(paper_ids, conn)

    Wlist = []

    #
    # Rules 10a, b, c, d, e
    #
    cited_paper_table = pd.merge(
        author_paper_table.reset_index()[["paper_id", "index"]].rename(
            columns={"paper_id": "target"}
        ),
        citation_table,
        on="target",
        how="left",
    ).dropna()
    U, _, _ = to_binary_matrix(
        cited_paper_table["index"].values,
        cited_paper_table["source"].values,
        num_rows=author_paper_table.shape[0],
    )
    bib_coupling_count_mat = U @ U.T
    bib_coupling_count_mat.data = np.maximum(2 * bib_coupling_count_mat.data, 10)
    Wlist += [bib_coupling_count_mat]

    #
    # Rules 11a, b, c, d, e
    #
    citing_paper_table = pd.merge(
        author_paper_table.reset_index()[["paper_id", "index"]].rename(
            columns={"paper_id": "source"}
        ),
        citation_table,
        on="source",
        how="left",
    ).dropna()
    U, _, _ = to_binary_matrix(
        citing_paper_table["index"].values,
        citing_paper_table["target"].values,
        num_rows=author_paper_table.shape[0],
    )
    co_citation_count_mat = U @ U.T
    co_citation_count_mat.data = np.maximum(co_citation_count_mat.data + 1, 6)
    Wlist += [co_citation_count_mat]

    #
    # Rules 9
    #
    citing_paper_table = citing_paper_table.iloc[
        np.isin(citing_paper_table["target"].values, paper_ids)
    ]
    node_by_paper, _, _ = to_binary_matrix(
        citing_paper_table["index"].values,
        citing_paper_table["target"].values,
        num_rows=author_paper_table.shape[0],
    )
    W = node_by_paper @ node_by_paper.T
    W.data = np.ones_like(W.data) * 10
    Wlist += [W]

    W = np.sum(Wlist, axis=0).toarray()

    return W
