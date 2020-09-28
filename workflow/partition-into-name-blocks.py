#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import sqlite3
from scipy import sparse
import os
import sys

if __name__ == "__main__":

    WOS_DISAMBIGUATION_DB = sys.argv[1]  # "../data/wos-disambiguation-data.db"
    OUTPUT = sys.argv[2]  # "../data/name-blocks"

    conn = sqlite3.connect(WOS_DISAMBIGUATION_DB)
    cur = conn.cursor()
    name_table = pd.read_sql("select * from name_table", conn)
    name_paper_table = pd.read_sql("select * from name_paper_table", conn)
    conn.close()

    name_paper_table = pd.merge(name_paper_table, name_table, on="name_id", how="left")

    name_paper_table[["block_id", "paper_id"]].drop_duplicates().to_csv(OUTPUT, index=False)
