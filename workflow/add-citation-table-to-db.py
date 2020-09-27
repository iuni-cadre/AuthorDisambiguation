#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import sqlite3
from scipy import sparse
import os
import sys


# # Data source
if __name__ == "__main__":

    WOS_UID_FILE = sys.argv[1]
    WOS_CITATION_DB = sys.argv[2]  # "../data/wos-citation.db"
    INPUT_DB = sys.argv[3]  # "../data/wos-disambiguation-data.db"
    OUTPUT_DB = sys.argv[4]  # "../data/wos-disambiguation-data.db"

    paper_table = pd.read_csv(WOS_UID_FILE)
    wos_ids = paper_table["UID"].drop_duplicates().values

    conn = sqlite3.connect(WOS_CITATION_DB)
    citation_table = pd.read_sql(
        "select citing as source, cited as target from citation_table where source in ({wos_ids}) or target in ({wos_ids})".format(
            wos_ids=",".join(['"%s"' % s for s in wos_ids])
        ),
        conn,
    )

    #if os.path.exists(OUTPUT_DB):
    #    os.remove(OUTPUT_DB)

    conn = sqlite3.connect(INPUT_DB)
    cur = conn.cursor()

    pd.DataFrame(citation_table).to_sql(
        "citation_table", conn, if_exists="append", index=False
    )
