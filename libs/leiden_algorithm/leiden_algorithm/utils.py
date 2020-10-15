from scipy import sparse
import networkx as nx
import numpy as np
import pandas as pd
import requests
import json

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


def get_connected_component(W):
    if isinstance(W, np.ndarray):
        G = nx.from_numpy_matrix(W)
    else:
        G = nx.from_scipy_sparse_matrix(W)
    comps = [
        np.array(list(c))
        for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]
    cids = np.zeros(W.shape[0], dtype=int)
    for c, comp in enumerate(comps):
        cids[comp] = c
    return cids


def slice_columns(tb, cols):
    tb = tb.copy()
    for c in cols:
        if c not in tb.columns:
            tb[c] = None

    return tb[cols]

#
# Retrieve the bibliographics from the Elastic Search
#
def find_papers_by_UID(uri, uids, max_request_records=1000):
    def find_papers_by_UID(uri, uids):
        """Simple Elasticsearch Query"""
        query = json.dumps({"query": {"ids": {"values": uids}}, "size": len(uids),})
        headers = {"Content-Type": "application/json"}
        response = requests.get(uri, headers=headers, data=query)
        results = json.loads(response.text)
        return results

    num_rounds = np.ceil(len(uids) / max_request_records).astype(int)
    all_results = []
    for i in range(num_rounds):
        sidx = max_request_records * i
        fidx = sidx + max_request_records
        results = find_papers_by_UID(uri, uids[sidx:fidx])
        all_results += results["hits"]["hits"]
    return all_results
