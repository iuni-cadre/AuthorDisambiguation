#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path
import json

sys.path.append(os.path.abspath(os.path.join("libs/WOS")))
import WOS


# In[9]:

if __name__ == "__main__":

    # dataPath = Path() #Path("../libs")
    WOSDataPath = Path(sys.argv[1])  # dataPath / "WOSData.bgz"
    UID2PositionPath = Path(sys.argv[2])  # dataPath / "UID2Positions.bgz"
    vincent_data_file = sys.argv[
        3
    ]  # "../data/sampled-disambiguationBenchmarkLabels.csv"
    OUTPUT = sys.argv[4]

    #
    # Loading
    #
    paper_table = pd.read_csv(vincent_data_file)
    UID2Positions = WOS.readIndicesDictionary(UID2PositionPath)

    #
    # Get WOS IDS
    #
    WOS_IDS = paper_table["WoSid"].values

    #
    # Retrieving json files
    #
    reader = WOS.DatabaseReader(WOSDataPath)
    paper_list = []
    for ID in WOS_IDS:
        paper_list += [reader.articleAt(UID2Positions[ID])]

    #
    # Output
    #
    with open(OUTPUT, "w") as f:
        for i, json_data in enumerate(paper_list):
            if i == 0:
                f.write(json.dumps(json_data))
            else:
                f.write("\n" + json.dumps(json_data))
        f.close()
