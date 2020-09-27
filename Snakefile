from os.path import join as j
import numpy as np
import glob

configfile: "workflow/config.yaml"

ES_PASSWORD = config["es_password"]
ES_USERNAME = config["es_username"]
ES_ENDPOINT = config["es_endpoint"]

WOS_CITATION_FILE = "/gpfs/sciencegenome/WoSjson/WoSedges/citeEdges.csv.gz"
WOS_CITATION_DB = "data/wos-citation.db"
WOS_DISAMBIGUATION_DB = "data/wos-disambiguation-data.db"

WOS_UID_FILE = "data/testData.csv"

rule all:
    input: WOS_CITATION_DB 

rule construct_wos_citation_db:
    input: WOS_CITATION_FILE
    output: WOS_CITATION_DB
    run:
        shell("python workflow/csv2sqlite.py --gzip {input} {output} && bash workflow/indexing-sql-table.sh {output}")

rule construct_disambiguation_paper_db:
    input: WOS_UID_FILE, WOS_CITATION_DB
    output: WOS_DISAMBIGUATION_DB 
    run:
        shell("python workflow/make-sqldb.py {input} {ES_PASSWORD} {ES_USERNAME} {ES_ENDPOINT} {output}")
