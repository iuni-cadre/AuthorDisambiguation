import itertools
import numpy as np
import pandas as pd
import sqlite3
import requests
import json
import os
import shutil
from tqdm import tqdm


class DataBlockingAlgorithm:

    def __init__(self, ES_USERNAME, ES_PASSWORD, ES_ENDPOINT, CITATION_DB):
        self.es_end_point = "http://{user}:{password}@{endpoint}".format(
            user=ES_USERNAME, password=ES_PASSWORD, endpoint=ES_ENDPOINT
        )
        self.conn = sqlite3.connect(CITATION_DB)

    def run(self, wos_ids, output_dir, writing_mode="append"):

        # Retrieve data from Elastic search
        results = self.find_papers_by_UID(self.es_end_point, wos_ids)

        #
        # Parse
        #
        address_table = self.parse_address_name(results)
        author_table = self.parse_author_name(results)
        paper_info = self.parse_paper_info(results)
        grant_table = self.parse_grant_name(results)

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
            block_table["block_id"] = np.arange(block_table.shape[0]).astype(int)
            return block_table

        def construct_name_table(author_block, table_block):
            name_table = pd.merge(
                author_table, block_table, on="normalized_name", how="left"
            )
            name_table = name_table[
                [
                    "name",
                    "initials",
                    "first_name",
                    "last_name",
                    "normalized_name",
                    "block_id",
                ]
            ].drop_duplicates()

            # Normalize
            name_table["first_name"] = name_table["first_name"].str.replace(
                "[^a-zA-Z ]", ""
            )
            name_table["last_name"] = name_table["last_name"].str.replace(
                "[^a-zA-Z ]", ""
            )
            name_table["name_id"] = np.arange(name_table.shape[0])

            # add short name
            name_table["short_name"] = (
                name_table["last_name"].apply(
                    lambda x: x.lower() if isinstance(x, str) else ""
                )
                + "_"
                + name_table["first_name"].apply(
                    lambda x: self.get_first_char(x).lower()
                )
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
                address_table,
                on=["paper_id", "_addr_no"],
                how="left",
            )
            return name_paper_addr_table

        block_table = construct_block_table(author_table)
        name_table = construct_name_table(author_table, block_table)
        name_paper_table = construct_name_paper_table(author_table, name_table)
        name_paper_address_table = construct_name_paper_address_table(
            name_paper_table, address_table
        )

        # generate paper_address_table
        paper_address_table = address_table[pd.isna(address_table["_addr_no"])]

        # (paper_table)
        paper_table = paper_info.copy().rename(columns={"source": "journal"})

        #
        # Save the generated tables
        #
        if writing_mode == "w":
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        for _, block in tqdm(
            block_table.iterrows(),
            total=block_table.shape[0],
            desc="Blocking by author names",
        ):
            block_name = block["normalized_name"]
            block_id = block["block_id"]

            sub_name_paper_table = name_paper_table[
                name_paper_table.block_id == block_id
            ]
            sub_name_table = name_table[name_table.block_id == block_id]
            sub_block_table = block_table[block_table.block_id == block_id]
            sub_paper_table = pd.merge(
                sub_name_paper_table[["paper_id"]], paper_table, on="paper_id"
            )
            sub_name_paper_address_table = pd.merge(
                name_paper_address_table[["name_paper_id"]],
                name_paper_address_table,
                on=["name_paper_id"],
            )
            sub_paper_address_table = pd.merge(
                paper_address_table[["paper_id"]], paper_address_table, on=["paper_id"]
            )
            sub_grant_table = pd.merge(
                paper_address_table[["paper_id"]], grant_table, on=["paper_id"]
            )

            # Load the citation table
            sub_citation_table = pd.read_sql(
                "select citing as source, cited as target from citation_table where source in ({wos_ids}) or target in ({wos_ids})".format(
                    wos_ids=",".join(['"%s"' % s for s in wos_ids])
                ),
                self.conn,
            ).dropna()

            #
            # Output to SQL
            #
            output_db = "%s/%s.db" % (output_dir, block_name)
            if os.path.exists(output_db):
                os.remove(output_db)
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

    #
    # Retrieve the bibliographics from the UID
    #
    def find_papers_by_UID(self, uri, uids, max_request_records=1000):
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
    # Decorator for Database class
    def safe_parse(parse_func):
        def wrapper(self, results, *args, **kwargs):
            df_list = []
            for result in results:
                UID = result["_id"]
                df = parse_func(self, result, *args, **kwargs)
                if df is None:
                    continue
                df["paper_id"] = UID
                df_list += [df]
            df = pd.concat(df_list, ignore_index=True)
            return df

        return wrapper

    @safe_parse
    def parse_grant_name(self, result):
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
                if len(grant_id) == 0:
                    merged += [None]
                else:
                    merged += [grant_id]
        df = pd.DataFrame(merged, columns=["grant_id"])
        return df

    @safe_parse
    def parse_address_name(self, result):
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

        address_name = result["_source"]["doc"].get("address_name", [])
        merged = [r["address_spec"] for r in list(itertools.chain(*address_name))]
        df = pd.DataFrame(merged)

        if df.shape[0] == 0:
            return None

        if "suborganizations" not in df.columns:
            df["suborganizations"] = None
        df = df.rename(
            columns={"organizations": "organization", "suborganizations": "department",}
        )
        df = df[
            [
                "full_address",
                "city",
                "country",
                "organization",
                "department",
                "_addr_no",
            ]
        ]
        df["organization"] = df["organization"].apply(
            lambda x: get_pref_records(x.get("organization", []))["_VALUE"]
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
    def parse_author_name(self, result):
        def get_initials(first_name, last_name):
            return self.get_first_char(first_name) + self.get_first_char(last_name)

        def get_normalized_name(first_name, last_name):
            def get_name(x, default=""):
                if isinstance(x, str):
                    return x.lower()
                else:
                    return default

            return get_name(last_name) + get_name(self.get_first_char(first_name))

        author_name = result["_source"]["doc"].get("name", [])
        df = pd.DataFrame([r for r in list(itertools.chain(*author_name))])

        # Create the normalized name and initials
        df = df.rename(columns={"wos_standard": "name"})
        df["initials"] = df.apply(
            lambda x: get_initials(x["first_name"], x["last_name"]), axis=1
        )
        df["normalized_name"] = df.apply(
            lambda x: get_normalized_name(x["first_name"], x["last_name"]), axis=1
        )
        return df

    @safe_parse
    def parse_paper_info(self, result):
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

    def get_first_char(self, x, default=""):
        if isinstance(x, str):
            return x[0]
        else:
            return default
