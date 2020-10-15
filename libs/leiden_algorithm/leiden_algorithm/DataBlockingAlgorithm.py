import itertools

# import numpy as np
import pandas as pd
import sqlite3

# import json
import os
import shutil
import hashlib
from tqdm import tqdm
from .utils import *
from joblib import Parallel, delayed


class DataBlockingAlgorithm:
    def __init__(self, CITATION_DB, n_jobs=30):
        #def __init__(self, ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB, n_jobs=30):
        #self.es_end_point = "http://{user}:{password}@{endpoint}".format(
        #    user=ES_USERNAME, password=ES_PASSWORD, endpoint=ES_ENDPOINT
        #)
        self.conn = sqlite3.connect(CITATION_DB)
        self.CITATION_DB = CITATION_DB
        self.n_jobs = n_jobs

    def run(self, JSON_RECORDS, output_dir, writing_mode="append"):

        def json2table(json_records):
            #
            # Make name_table and block_table
            #
            def construct_block_table(author_table):
                block_table = (
                    author_table[["normalized_name"]]
                    .drop_duplicates()
                    .reset_index()
                    .drop(columns=["index"])
                )
                block_table["block_id"] = block_table.apply(
                    lambda x: hashlib.sha256(str.encode(x["normalized_name"])).hexdigest(),
                    axis=1,
                )
                return block_table
    
            def construct_name_table(author_block, table_block):
                name_table = pd.merge(
                    author_table, block_table, on="normalized_name", how="left"
                )
                name_table = slice_columns(
                    name_table,
                    [
                        "name",
                        "initials",
                        "first_name",
                        "last_name",
                        "normalized_name",
                        "block_id",
                    ],
                ).drop_duplicates()
    
                # Normalize
                name_table["first_name"] = name_table["first_name"].str.replace(
                    "[^a-zA-Z ]", ""
                )
                name_table["last_name"] = name_table["last_name"].str.replace(
                    "[^a-zA-Z ]", ""
                )
    
                def concat(x):
                    s = "_".join(["%s" % v for k, v in x.items()])
                    return str.encode(s)
    
                name_table["name_id"] = name_table.apply(
                    lambda x: hashlib.sha256(concat(x)).hexdigest(), axis=1
                )
    
                # add short name
                name_table["short_name"] = (
                    name_table["last_name"].apply(
                        lambda x: x.lower() if isinstance(x, str) else ""
                    )
                    + "_"
                    + name_table["first_name"].apply(lambda x: get_first_char(x).lower())
                )
                short_name_list = name_table["short_name"].drop_duplicates().values
                name_table = pd.merge(
                    name_table,
                    pd.DataFrame(
                        {
                            "short_name": short_name_list,
                            "short_name_id": [
                                hashlib.sha256(str.encode(x)).hexdigest()
                                for x in short_name_list
                            ],
                        }
                    ),
                    on="short_name",
                    how="left",
                )
                return name_table
    
            def construct_name_paper_table(author_table, name_table):
    
                name_paper_table = author_table.copy().rename(
                    columns={
                        "wos_standard": "name",
                        "UID": "paper_id",
                        "email_addr": "email_address",
                    }
                )
                
                name_paper_table = pd.merge(
                    name_paper_table, name_table[["name_id", "name"]], on="name", how="left"
                ).drop(columns="name")
    
                if "_addr_no" is not None:
                    name_paper_table["_addr_no"] = None
                name_paper_table = slice_columns(
                    name_paper_table,
                    ["name_id", "paper_id", "email_address", "_addr_no", "_seq_no"],
                )
    
                def concat(x):
                    s = "_".join(["%s" % v for k, v in x.items()])
                    return str.encode(s)
    
                name_paper_table["name_paper_id"] = name_paper_table.apply(
                    lambda x: hashlib.sha256(concat(x)).hexdigest(), axis=1
                )
                name_paper_table = pd.merge(
                    name_paper_table,
                    name_table[["name_id", "short_name_id", "block_id"]],
                    on="name_id",
                    how="left",
                )
                return name_paper_table
    
            def construct_name_paper_address_table(name_paper_table, address_table):
    
                # generate name_paper_address_table
                #
                # author-paper affiliation table
                name_paper_addr_table = name_paper_table.copy()[
                    ["paper_id", "_addr_no", "name_paper_id"]
                ].dropna()
                name_paper_addr_list = []
                for i, row in name_paper_addr_table.iterrows():
                    name_paper_addr_list += [
                        (row["name_paper_id"], row["paper_id"], int(_addr_no))
                        for _addr_no in row["_addr_no"].split(" ")
                    ]
                name_paper_addr_table = pd.DataFrame(
                    name_paper_addr_list, columns=["name_paper_id", "paper_id", "_addr_no"]
                )
                name_paper_addr_table = pd.merge(
                    name_paper_addr_table[["name_paper_id", "paper_id", "_addr_no"]],
                    address_table.rename(columns = {"UID": "paper_id"}),
                    on=["paper_id", "_addr_no"],
                    how="left",
                )
                return name_paper_addr_table
            
             
            #
            # Parse
            #
            address_table = parse_address_name(json_records)
            author_table = parse_author_name(json_records)
            paper_info = parse_paper_info(json_records)
            grant_table = parse_grant_name(json_records)
    
            block_table = construct_block_table(author_table)
            name_table = construct_name_table(author_table, block_table)
            name_paper_table = construct_name_paper_table(author_table, name_table)
            name_paper_address_table = construct_name_paper_address_table(
                name_paper_table, address_table
            )
    
            # generate paper_address_table
            paper_address_table = address_table[pd.isna(address_table["_addr_no"])]
    
            # (paper_table)
            paper_table = paper_info.copy().rename(columns={"source": "journal", "UID":"paper_id"})
    
            return block_table, name_paper_table, name_table, paper_table, name_paper_address_table, paper_address_table, grant_table

        def export_block_to_sql(
            block,
            name_paper_table,
            name_table,
            block_table,
            paper_table,
            name_paper_address_table,
            paper_address_table,
            grant_table,
            CITATION_DB,
            output_dir,
        ):

            block_name = block["normalized_name"]
            block_id = block["block_id"]

            sub_name_paper_table = name_paper_table[
                name_paper_table.block_id == block_id
            ].drop_duplicates()
            sub_name_table = name_table[
                name_table.block_id == block_id
            ].drop_duplicates()
            sub_block_table = block_table[
                block_table.block_id == block_id
            ].drop_duplicates()
            sub_paper_table = pd.merge(
                sub_name_paper_table[["paper_id"]], paper_table, on="paper_id"
            ).drop_duplicates()
            sub_name_paper_address_table = pd.merge(
                name_paper_address_table[["name_paper_id"]],
                name_paper_address_table,
                on=["name_paper_id"],
            ).drop_duplicates()
            sub_paper_address_table = pd.merge(
                paper_address_table[["paper_id"]], paper_address_table, on=["paper_id"]
            ).drop_duplicates()
            sub_grant_table = pd.merge(
                paper_address_table[["paper_id"]], grant_table, on=["paper_id"]
            ).drop_duplicates()

            # Load the citation table
            wos_ids = sub_paper_table.paper_id.drop_duplicates().values
            sub_citation_table = pd.read_sql(
                "select citing as source, cited as target from citation_table where source in ({wos_ids}) or target in ({wos_ids})".format(
                    wos_ids=",".join(['"%s"' % s for s in wos_ids])
                ),
                sqlite3.connect(CITATION_DB),
            ).dropna()

            #
            # Output to SQL
            #
            output_db = "%s/%s.db" % (output_dir, block_name)
            # if os.path.exists(output_db):
            #    os.remove(output_db)
            sub_conn = sqlite3.connect(output_db)

            sub_name_paper_table.to_sql(
                "name_paper_table", sub_conn, if_exists="append", index=False
            )

            sub_name_table.to_sql(
                "name_table", sub_conn, if_exists="append", index=False
            )

            sub_block_table.to_sql(
                "block_table", sub_conn, if_exists="append", index=False
            )

            sub_paper_table.to_sql(
                "paper_table", sub_conn, if_exists="append", index=False
            )

            sub_name_paper_address_table.to_sql(
                "name_paper_address_table", sub_conn, if_exists="append", index=False
            )

            sub_paper_address_table.to_sql(
                "paper_address_table", sub_conn, if_exists="append", index=False
            )
            sub_grant_table.to_sql(
                "grant_table", sub_conn, if_exists="append", index=False
            )

            sub_citation_table.to_sql(
                "citation_table", sub_conn, if_exists="append", index=False
            )
            sub_conn.close()

        num_files = len(JSON_RECORDS)
        num_chunks = np.ceil(num_files / self.n_jobs)
        block_table = [] 
        name_paper_table = []
        name_table = []
        paper_table = []
        name_paper_address_table = []
        paper_address_table = []
        grant_table = []
        records = Parallel(n_jobs=self.n_jobs)(
            delayed(json2table)([JSON_RECORDS[i] for i in chunks]) for chunks in tqdm(np.array_split(np.arange(num_files), num_chunks))
        )

        for record in records:
            block_table+=[record[0]]
            name_paper_table+=[record[1]]
            name_table+=[record[2]]
            paper_table+=[record[3]]
            name_paper_address_table+=[record[4]]
            paper_address_table+=[record[5]]
            grant_table+=[record[6]]
        
        block_table = pd.concat(block_table, ignore_index = True) 
        name_paper_table = pd.concat(name_paper_table, ignore_index = True)
        name_table = pd.concat(name_table, ignore_index = True)
        paper_table = pd.concat(paper_table, ignore_index = True)
        name_paper_address_table = pd.concat(name_paper_address_table, ignore_index = True)
        paper_address_table = pd.concat(paper_address_table, ignore_index = True)
        grant_table = pd.concat(grant_table, ignore_index = True)
         
        #
        # Save the generated tables
        #
        if writing_mode == "w":
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)


        Parallel(n_jobs=self.n_jobs)(
            delayed(export_block_to_sql)(
                block,
                name_paper_table,
                name_table,
                block_table,
                paper_table,
                name_paper_address_table,
                paper_address_table,
                grant_table,
                self.CITATION_DB,
                output_dir,
            )
            for _, block in block_table.iterrows()
        )
#
# Parse the retrieved json and store them as pandas table
#
# Decorator for Database class
#def safe_parse(parse_func):
#    def wrapper(results, n_jobs=1, *args, **kwargs):
#        df_list = []
#
#        def func(result):
#            UID = result["UID"]
#            df = parse_func(result, *args, **kwargs)
#            if df is None:
#                return None
#            df["paper_id"] = UID
#            return df
#
#        df_list = Parallel(n_jobs=n_jobs)(delayed(func)(result) for result in results)
#        df_list = [df for df in df_list if df is not None]
#        df = pd.concat(df_list, ignore_index=True)
#        return df
#
#    return wrapper

def safe_parse(parse_func):
    def wrapper(results, *args, **kwargs):
        df_list = []

        def func(result):
            UID = result["UID"]
            df = parse_func(result, *args, **kwargs)
            if df is None:
                return None
            df["paper_id"] = UID
            return df
        df_list = [func(result) for result in results]
        df_list = [df for df in df_list if df is not None]
        df = pd.concat(df_list, ignore_index=True)
        return df

    return wrapper


@safe_parse
def parse_grant_name(result):
    fund_ack = [result.get("fund_ack", {"grants": {"grant": []}})]
    grants = [r["grants"]["grant"] for r in fund_ack]
    grant_ids = [
        r["grant_ids"]["grant_id"]
        for r in list(itertools.chain(*grants))
        if "grant_ids" in r
    ]
    merged = []
    for grant_id in grant_ids:
        if isinstance(grant_id, list):
            merged += grant_id
        else:
            if len(grant_id) == 0:
                merged += [None]
            else:
                merged += [grant_id]
    df = pd.DataFrame(merged, columns=["grant_id"])
    return df


@safe_parse
def parse_address_name(result):
    def get_pref_records(records):
        if len(records) == 0:
            return None
        for record in records:
            if "_pref" in record:
                return record
        return records[0]

    def get_department(dept):
        if isinstance(dept, list):
            return dept[0]
        return dept

    address_name = result.get("address_name", [])
    merged = [r["address_spec"] for r in address_name]
    #merged = [r["address_spec"] for r in list(itertools.chain(*address_name))]
    
    df = pd.DataFrame(merged)

    if df.shape[0] == 0:
        return None

    if "suborganizations" not in df.columns:
        df["suborganizations"] = None
    if "organizations" not in df.columns:
        df["organizations"] = None
    if "city" not in df.columns:
        df["city"] = None
    if "country" not in df.columns:
        df["country"] = None
    df = df.rename(
        columns={"organizations": "organization", "suborganizations": "department",}
    )
    df = df[
        ["full_address", "city", "country", "organization", "department", "_addr_no",]
    ]
    df["organization"] = df["organization"].apply(
        lambda x: get_pref_records(x.get("organization", []))["_VALUE"]
        if isinstance(x, dict)
        else None
    )
    df["department"] = (
        df["department"]
        .apply(
            lambda x: get_department(x["suborganization"])
            if isinstance(x, dict)
            else None
        )
        .astype(str)
    )
    return df


@safe_parse
def parse_author_name(result):
    def get_initials(first_name, last_name):
        return get_first_char(first_name) + get_first_char(last_name)

    def get_normalized_name(first_name, last_name):
        def get_name(x, default=""):
            if isinstance(x, str):
                return x.lower()
            else:
                return default

        return get_name(last_name) + get_name(get_first_char(first_name))

    author_name = result.get("name", [])
    df = pd.DataFrame(author_name)
    #df = pd.DataFrame([r for r in list(itertools.chain(*author_name))])

    # Create the normalized name and initials
    df = df.rename(columns={"wos_standard": "name"})
    df["initials"] = df.apply(
        lambda x: get_initials(x.get("first_name", ""), x.get("last_name", "")), axis=1
    )
    df["normalized_name"] = df.apply(
        lambda x: get_normalized_name(x.get("first_name", ""), x.get("last_name", "")),
        axis=1,
    )
    return df


@safe_parse
def parse_paper_info(result):
    #
    # Publication year
    #
    pub_info = result.get("pub_info", [])
    if len(pub_info) >= 1:
        pub_year = pub_info.get("_pubyear", float("NaN"))
    else:
        pub_year = float("NaN")

    #
    # Titles and source
    #
    titles = result.get("titles", [])
    if len(titles) > 0:
        titles = titles.get("title", [])
        title = ""
        source = ""
        source_iso = ""
        for r in titles:
            if r["_type"] == "source":
                source = r["_VALUE"]
            elif r["_type"] == "abbrev_iso":
                source_iso = r["_VALUE"]
            elif r["_type"] == "item":
                title = r["_VALUE"]

    #
    # Grant number not implemented
    #
    df = pd.DataFrame(
        [
            {
                "source": source,
                "title": title,
                "source_iso": source_iso,
                "pub_year": pub_year,
            }
        ]
    )
    return df


def get_first_char(x, default=""):
    if isinstance(x, str):
        if len(x) == 0:
            return ""
        return x[0]
    else:
        return default
