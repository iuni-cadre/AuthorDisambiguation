import pandas as pd
import sqlite3
import sys
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
    conn = sqlite3.connect(OUTPUT_SQLFILE)

    def to_dataframe(filename):
        records = []
        for line in open(filename, "r"):
            record = json.loads(line)
            UID = record["UID"]
            records += [{"UID": UID, "JSON": json.dumps(record)}]
        df = pd.DataFrame(records)
        return df

    df_list = Parallel(n_jobs=n_jobs)(
        delayed(to_dataframe)(JSON_FILE) for JSON_FILE in tqdm(JSON_FILES)
    )

    df = pd.concat(df_list)
    df.to_sql("json_table", conn, if_exists="replace")
    conn.close()
