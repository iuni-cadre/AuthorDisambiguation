from os.path import join as j
import numpy as np
import glob

configfile: "workflow/config.yaml"


WOS_CITATION_DB = "data/wos-citation.db"
WOS_CITATION_FILE = "/gpfs/sciencegenome/WoSjson/WoSedges/citeEdges.csv.gz"



rule all:
    input: WOS_CITATION_DB 

rule construct_wos_citation_db:
    input: WOS_CITATION_FILE
    output: WOS_CITATION_DB
    run:
        shell("python workflow/csv2sqlite.py --gzip {input} {output}")

