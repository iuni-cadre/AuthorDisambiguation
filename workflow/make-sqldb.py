#!/usr/bin/env python
# coding: utf-8
import itertools
import numpy as np
import pandas as pd
import sqlite3
import requests
import json
import os
import sys

#
# Retrieve the bibliographics from the UID
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


#
# Parse the retrieved json and store them as pandas table
#
def safe_parse(parse_func):
    def wrapper(results, *args, **kwargs):
        df_list = []
        for result in results:
            UID = result["_id"]

            df = parse_func(result, *args, **kwargs)

            df["UID"] = UID
            df_list += [df]
        df = pd.concat(df_list, ignore_index=True)
        return df

    return wrapper


@safe_parse
def parse_grant_name(result):
    fund_ack = result["_source"]["doc"].get("fund_ack", [{"grants": {"grant": []}}])
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
            merged += [grant_id]
    df = pd.DataFrame(merged, columns=["grant_id"])
    return df


@safe_parse
def parse_address_name(result):
    address_name = result["_source"]["doc"].get("address_name", [])
    merged = [r["address_spec"] for r in list(itertools.chain(*address_name))]
    df = pd.DataFrame(merged)
    return df


@safe_parse
def parse_author_name(result):
    author_name = result["_source"]["doc"].get("name", [])
    df = pd.DataFrame([r for r in list(itertools.chain(*author_name))])
    return df


@safe_parse
def parse_paper_info(result):
    #
    # Publication year
    #
    pub_info = result["_source"]["doc"].get("pub_info", [])
    if len(pub_info) >= 1:
        pub_year = pub_info[0].get("_pubyear", float("NaN"))
    else:
        pub_year = float("NaN")

    #
    # Titles and source
    #
    titles = result["_source"]["doc"].get("titles", [])
    if len(titles) > 0:
        titles = titles[0].get("title", [])
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
    grant_number = ""
    df = pd.DataFrame(
        [
            {
                "source": source,
                "title": title,
                "source_iso": source_iso,
                "pub_year": pub_year,
                "grant_number": grant_number,
            }
        ]
    )
    return df


#
# For Names
#
def get_first_char(x, default=""):
    if isinstance(x, str):
        return x[0]
    else:
        return default


def get_initials(first_name, last_name):
    return get_first_char(first_name) + get_first_char(last_name)


def get_normalized_name(first_name, last_name):
    def get_name(x, default=""):
        if isinstance(x, str):
            return x.lower()
        else:
            return default

    return get_name(last_name) + get_name(get_first_char(first_name))


if __name__ == "__main__":

    WOS_ID_FILE = sys.argv[1]
    CITATION_DB = sys.argv[2]
    ES_PASSWORD = sys.argv[3]
    ES_USERNAME = sys.argv[4]
    ES_ENDPOINT = sys.argv[5]
    OUTPUT_DB = sys.argv[6]

    # Retrieve the wos_ids
    wos_ids = pd.read_csv(WOS_ID_FILE)["UID"].drop_duplicates().values.tolist()

    #
    # Retrieve the citation data from the WOS
    #
    conn = sqlite3.connect(CITATION_DB)
    citation_table = pd.read_sql(
        "select citing as source, cited as target from citation_table where source in ({wos_ids}) or target in ({wos_ids})".format(
            wos_ids=",".join(['"%s"' % s for s in wos_ids])
        ),
        conn,
    )

    #
    # Retrieve the bibliographic data from WOS database
    #
    es_end_point = "http://{user}:{password}@{endpoint}".format(
        user=ES_USERNAME, password=ES_PASSWORD, endpoint=ES_ENDPOINT
    )

    citation_table = citation_table.dropna()
    results = find_papers_by_UID(es_end_point, wos_ids)

    #
    # Parse
    #
    address_table = parse_address_name(results)
    author_table = parse_author_name(results)
    paper_info = parse_paper_info(results)
    grant_table = parse_grant_name(results)

    #
    # Make name_table and block_table
    #
    # (name_table, block_table)

    # Create the normalized name and initials
    author_table = author_table.rename(
        columns={"wos_standard": "name", "UID": "paper_id"}
    )
    author_table["initials"] = author_table.apply(
        lambda x: get_initials(x["first_name"], x["last_name"]), axis=1
    )
    author_table["normalized_name"] = author_table.apply(
        lambda x: get_normalized_name(x["first_name"], x["last_name"]), axis=1
    )

    # Create block table
    block_table = (
        author_table[["normalized_name"]]
        .drop_duplicates()
        .reset_index()
        .drop(columns=["index"])
    )
    block_table["block_id"] = np.arange(block_table.shape[0]).astype(int)
    name_table = pd.merge(author_table, block_table, on="normalized_name", how="left")
    name_table = name_table[
        ["name", "initials", "first_name", "last_name", "normalized_name", "block_id"]
    ].drop_duplicates()

    # Normalize
    name_table["first_name"] = name_table["first_name"].str.replace("[^a-zA-Z ]", "")
    name_table["last_name"] = name_table["last_name"].str.replace("[^a-zA-Z ]", "")
    name_table["name_id"] = np.arange(name_table.shape[0])

    # add short name
    name_table["short_name"] = (
        name_table["last_name"].apply(lambda x: x.lower() if isinstance(x, str) else "")
        + "_"
        + name_table["first_name"].apply(lambda x: get_first_char(x).lower())
    )
    short_name_list = name_table["short_name"].drop_duplicates().values
    name_table = pd.merge(
        name_table,
        pd.DataFrame(
            {
                "short_name": short_name_list,
                "short_name_id": np.arange(short_name_list.size),
            }
        ),
        on="short_name",
        how="left",
    )

    #
    # Make name_paper_table
    #
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
    name_paper_table = name_paper_table[
        ["name_id", "paper_id", "email_address", "_addr_no"]
    ]
    name_paper_table["name_paper_id"] = np.arange(name_paper_table.shape[0])
    name_paper_table = pd.merge(
        name_paper_table,
        name_table[["name_id", "short_name_id", "block_id"]],
        on="name_id",
        how="left",
    )

    #
    # paper_address_table
    #
    # Make address table that contain both author-associated and non-associated addresses
    address_table = address_table.rename(
        columns={
            "organizations": "organization",
            "suborganizations": "department",
            "UID": "paper_id",
        }
    )
    address_table = address_table[
        [
            "paper_id",
            "full_address",
            "city",
            "country",
            "organization",
            "department",
            "_addr_no",
        ]
    ]

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

    address_table["organization"] = address_table["organization"].apply(
        lambda x: get_pref_records(x.get("organization", []))["_VALUE"]
    )
    address_table["department"] = (
        address_table["department"]
        .apply(
            lambda x: get_department(x["suborganization"])
            if isinstance(x, dict)
            else None
        )
        .astype(str)
    )

    #
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
        address_table,
        on=["paper_id", "_addr_no"],
        how="left",
    )

    # generate paper_address_table
    paper_address_table = address_table[pd.isna(address_table["_addr_no"])]

    # (paper_table)
    paper_table = paper_info.copy()
    paper_table = paper_table.rename(columns={"source": "journal", "UID": "paper_id"})

    # grant_table
    grant_table = grant_table.rename(columns={"UID": "paper_id"})

    #
    # Output to SQL
    #
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()

    pd.DataFrame(name_paper_table).to_sql(
        "name_paper_table", conn, if_exists="append", index=False
    )

    pd.DataFrame(name_table).to_sql("name_table", conn, if_exists="append", index=False)

    pd.DataFrame(block_table).to_sql(
        "block_table", conn, if_exists="append", index=False
    )

    pd.DataFrame(paper_table).to_sql(
        "paper_table", conn, if_exists="append", index=False
    )

    pd.DataFrame(name_paper_addr_table).to_sql(
        "name_paper_address_table", conn, if_exists="append", index=False
    )

    pd.DataFrame(paper_address_table).to_sql(
        "paper_address_table", conn, if_exists="append", index=False
    )
    pd.DataFrame(grant_table).to_sql(
        "grant_table", conn, if_exists="append", index=False
    )

    pd.DataFrame(citation_table).to_sql(
        "citation_table", conn, if_exists="append", index=False
    )
    conn.close()
