from os.path import join as j
import numpy as np
import glob

configfile: "workflow/config.yaml"

ES_PASSWORD = config["es_password"]
ES_USERNAME = config["es_username"]
ES_ENDPOINT = config["es_endpoint"]

WOS_CITATION_FILE = "/gpfs/sciencegenome/WoSjson2019/citeEdges.csv/citeEdges.csv.gz"
WOS_CITATION_DB = "data/wos-citation.db"
WOS_DISAMBIGUATION_DB = "data/wos-disambiguation-data.db"

WOS_UID_FILE = "data/testData.csv"

# Blocked paper list
#BLOCK_PAPERS_LIST = "data/block-papers-list.csv"

# Author Name count file 
NAME_COUNT_FILE = "data/nameCount.csv"
GENERAL_NAME_LIST_FILE = "data/general-name-list.csv"
DISAMBIGUATION_WORKING_DIR = "data/disambiguation-working-dir"
DISAMBIGUATED_AUTHOR_LIST = "data/disambiguated-authors.csv"



rule all:
    input: DISAMBIGUATED_AUTHOR_LIST

rule construct_wos_citation_db:
    input: WOS_CITATION_FILE
    output: WOS_CITATION_DB
    run:
        shell("python workflow/csv2sqlite.py --gzip {input} {output} && bash workflow/indexing-sql-table.sh {output}")

rule make_general_name_list:
    input: NAME_COUNT_FILE
    output: GENERAL_NAME_LIST_FILE
    run:
        shell("python workflow/make-general-name-list.py {input} {output}")


rule disambiguation:
    input: WOS_UID_FILE, WOS_CITATION_DB, GENERAL_NAME_LIST_FILE
    output: DISAMBIGUATED_AUTHOR_LIST, directory(DISAMBIGUATION_WORKING_DIR)
    run:
        shell("python workflow/disambiguation.py {ES_USERNAME} {ES_PASSWORD} {ES_ENDPOINT} {WOS_UID_FILE} {WOS_CITATION_DB} {GENERAL_NAME_LIST_FILE} {DISAMBIGUATION_WORKING_DIR} {output}")
