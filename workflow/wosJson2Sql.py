import pandas as pd
import sqlite3
import numpy as np
import sys
import os
from joblib import Parallel, delayed
import json
from tqdm import tqdm
import glob

if __name__ == "__main__":

    JSON_DIR = sys.argv[1]
    OUTPUT_SQLFILE = sys.argv[
        2
    ]  # "/gpfs/sciencegenome/WoSjson2019/part-00459-44a9d770-0735-473f-8cda-fdf9a251fb94-c000.json"#sys.argv[1]
    n_jobs = 40

    JSON_FILES = glob.glob(JSON_DIR + "/*.json")

    if os.path.exists(OUTPUT_SQLFILE):
        os.remove(OUTPUT_SQLFILE)

    def to_dataframe(filename):
        records = []
        for line in open(filename, "r"):
            record = json.loads(line)
            UID = record["UID"]
            records += [{"UID": UID, "JSON": json.dumps(record)}]
        df = pd.DataFrame(records)
        return df

    num_files = len(JSON_FILES)
    num_chunks = np.ceil(num_files / n_jobs)
    for chunks in tqdm(np.array_split(np.arange(num_files), num_chunks)):

        conn = sqlite3.connect(OUTPUT_SQLFILE)

        df_list = Parallel(n_jobs=n_jobs)(
            delayed(to_dataframe)(JSON_FILES[i]) for i in chunks
        )
        df = pd.concat(df_list, ignore_index=True)
        df.to_sql("json_table", conn, if_exists="append", chunksize=500)

        conn.close()
