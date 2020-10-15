import numpy as np
import yaml
import pandas as pd
import requests
import json
import sys

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


if __name__ == "__main__":

    ES_CONFIG_FILE = sys.argv[1]
    INPUT_WOS_FILE = sys.argv[2]
    WOS_ID_COLUMN_NAME = sys.argv[3]
    OUTPUT = sys.argv[4]

    with open(ES_CONFIG_FILE) as f:
        ES = yaml.safe_load(f)
        ES_USERNAME = ES["es_username"]
        ES_PASSWORD = ES["es_password"]
        ES_ENDPOINT = ES["es_endpoint"]

    wos_ids = (
        pd.read_csv(INPUT_WOS_FILE)[WOS_ID_COLUMN_NAME]
        .drop_duplicates()
        .values.tolist()
    )

    es_end_point = "http://{user}:{password}@{endpoint}".format(
        user=ES_USERNAME, password=ES_PASSWORD, endpoint=ES_ENDPOINT
    )

    results = find_papers_by_UID(es_end_point, wos_ids)

    with open(OUTPUT, "w") as f:
        for i, record in enumerate(results):
            line = json.dumps(record)
            if i == 0:
                f.write(line)
            else:
                f.write("\n" + line)
