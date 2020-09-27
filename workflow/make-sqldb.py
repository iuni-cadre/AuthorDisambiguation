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
def find_papers_by_UID(uri, uids):
    """Simple Elasticsearch Query"""
    query = json.dumps({"query": {"ids": {"values": uids}}, "size": len(uids),})
    headers = {"Content-Type": "application/json"}
    response = requests.get(uri, headers=headers, data=query)
    results = json.loads(response.text)
    return results


#
# Parse the retrieved json and store them as pandas table
#
def safe_parse(parse_func):
    def wrapper(results, *args, **kwargs):
        df_list = []
        for result in results["hits"]["hits"]:
            UID = result["_id"]

            df = parse_func(result, *args, **kwargs)

            df["UID"] = UID
            df_list += [df]
        df = pd.concat(df_list, ignore_index=True)
        return df

    return wrapper


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
def get_initials(first_name, last_name):
    def get_first_char(x, default=""):
        if isinstance(x, str):
            return x[0]
        else:
            return default

    return get_first_char(first_name) + get_first_char(last_name)


def get_normalized_name(first_name, last_name):
    def get_name(x, default=""):
        if isinstance(x, str):
            return x.lower()
        else:
            return default

    def get_first_char(x, default=""):
        if isinstance(x, str):
            return x[0]
        else:
            return default

    return get_name(last_name) + get_name(get_first_char(first_name))


if __name__ == "__main__":

    WOS_ID_FILE = sys.argv[1]
    ES_PASSWORD = sys.argv[2]
    ES_USERNAME = sys.argv[3]
    ES_ENDPOINT = sys.argv[4]
    OUTPUT_DB = sys.argv[5]

    # Retrieve the wos_ids
    wos_ids = pd.read_csv(WOS_ID_FILE)["UID"].drop_duplicates().values

    #
    # Retrieve the bibliographic data from WOS database
    #
    es_end_point = "http://{user}:{password}@{endpoint}".format(
        user=ES_USERNAME, password=ES_PASSWORD, endpoint=ES_ENDPOINT
    )

    results = find_papers_by_UID(es_end_point, wos_ids.tolist())

    #
    # Parse
    #
    address_table = parse_address_name(results)
    author_table = parse_author_name(results)
    paper_info = parse_paper_info(results)

    #
    # Formatting
    #
    # (name_table, block_table)
    name_table = author_table.copy()
    name_table = name_table.rename(columns={"full_name": "name"})

    name_table["initials"] = name_table.apply(
        lambda x: get_initials(x["first_name"], x["last_name"]), axis=1
    )
    name_table["normalized_name"] = name_table.apply(
        lambda x: get_normalized_name(x["first_name"], x["last_name"]), axis=1
    )
    name_table = name_table[
        ["name", "initials", "first_name", "last_name", "normalized_name"]
    ].drop_duplicates()

    block_table = (
        name_table[["normalized_name"]]
        .drop_duplicates()
        .reset_index()
        .drop(columns=["index"])
    )
    block_table["block_id"] = np.arange(block_table.shape[0])

    name_table = pd.merge(name_table, block_table, on="normalized_name", how="left")
    name_table["name_id"] = np.arange(name_table.shape[0])

    # (paper_table)
    paper_table = paper_info.copy()
    paper_table = paper_table.rename(columns={"source": "journal", "UID": "paper_id"})

    # (name_paper_table)
    name_paper_table = author_table.copy().rename(
        columns={"full_name": "name", "UID": "paper_id", "email_addr": "email_address"}
    )
    name_paper_table = pd.merge(
        name_paper_table, name_table[["name_id", "name"]], on="name", how="left"
    ).drop(columns="name")
    name_paper_table = name_paper_table[["name_id", "paper_id", "email_address"]]

    # (address_table)
    address_table = address_table.rename(
        columns={"organizations": "organization", "suborganizations": "department"}
    )
    address_table = address_table[
        ["UID", "full_address", "city", "country", "organization", "department"]
    ]
    address_table["organization"] = address_table["organization"].astype(str)
    address_table["department"] = address_table["department"].astype(str)

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

    pd.DataFrame(address_table).to_sql(
        "address_table", conn, if_exists="append", index=False
    )
    conn.close()
