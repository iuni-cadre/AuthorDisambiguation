from os.path import join as j
import numpy as np
import glob
import glob


configfile: "workflow/config.yaml"


CONFIG_FILE = "workflow/config.yaml"
SHARED_DIR = config["shared_dir"]

# Files to construct the citation database
WOS_CITATION_FILE = "/gpfs/sciencegenome/WoSjson2019/citeEdges.csv/citeEdges.csv.gz"
WOS_CITATION_DB = j("data/", "wos-citation.db")

# Files to construct the database for json files
WOS_JSON_DIR = glob.glob("/gpfs/sciencegenome/WoSjson2019")
WOS_SQL_TABLE = j("data", "wos-json.db")

# Json file for WOS
WOS_JSON_DB = j("libs", "WOSData.bgz")
WOS_JSON_POS = j("libs", "UID2Positions.bgz")

# Author Name count file
NAME_COUNT_FILE = j(SHARED_DIR, "nameCount.csv")
GENERAL_NAME_LIST_FILE = j(SHARED_DIR, "general-name-list.csv")

# Input file
WOS_UID_FILE = j(SHARED_DIR, "disambiguationBenchmarkLabels.csv")
WOS_ID_COLUMN_NAME = "WoSid"
WOS_UID_FILE_SAMPLED = j("data", "sampled-disambiguationBenchmarkLabels.csv")
SAMPLE_NUM = 10000

WOS_JSON_FILE_SAMPLED = j("data", "sampled-disambiguationBenchmarkLabels.json")

# Input for the full disambiguation
#WOS_JSON_FILE_DIR = "data/sampled-json" # for testing
WOS_JSON_FILE_DIR = "/gpfs/sciencegenome/WoSjson2019"

# Working directory for the Leiden disambiguation algorithm
DISAMBIGUATION_WORKING_DIR = "data/disambiguation-working-dir"

# Results
DISAMBIGUATED_AUTHOR_LIST = "data/disambiguated-authors.csv"

# Validations
GROUND_TRUTH_AUTHOR_LIST = WOS_UID_FILE
DISAMBIGUATION_FOR_VALIDATION_TEST = "data/validation-disambiguated-authors.csv"
VALIDATION_RESULT = "data/result/validation-scores.txt"


rule all:
    input:
        DISAMBIGUATED_AUTHOR_LIST,


rule construct_wos_citation_db:
    input:
        WOS_CITATION_FILE,
    output:
        WOS_CITATION_DB,
    run:
        shell(
            "python workflow/csv2sqlite.py --gzip {input} {output} && bash workflow/indexing-sql-table.sh {output}"
        )


rule make_general_name_list:
    input:
        NAME_COUNT_FILE,
    output:
        GENERAL_NAME_LIST_FILE,
    run:
        shell("python workflow/make-general-name-list.py {input} {output}")


rule random_sampling_test_data:
    input:
        WOS_UID_FILE,
    output:
        WOS_UID_FILE_SAMPLED,
    run:
        shell(
            'cut -d"," -f 1 {input} |sed -s 1d |sort |uniq | shuf -n {SAMPLE_NUM} |sed -e "1iWoSid">{output} '
        )


rule full_disambiguation:
    input:
        #CONFIG_FILE,
        #WOS_UID_FILE_SAMPLED,
        WOS_JSON_FILE_DIR,
        WOS_CITATION_DB,
        GENERAL_NAME_LIST_FILE,
    output:
        DISAMBIGUATED_AUTHOR_LIST,
        directory(DISAMBIGUATION_WORKING_DIR),
    run:
        shell(
            "python workflow/disambiguation.py {WOS_JSON_FILE_DIR} {WOS_CITATION_DB} {GENERAL_NAME_LIST_FILE} {DISAMBIGUATION_WORKING_DIR} {output}"
            #"python workflow/disambiguation.py {CONFIG_FILE} {WOS_UID_FILE_SAMPLED} {WOS_ID_COLUMN_NAME} {WOS_CITATION_DB} {GENERAL_NAME_LIST_FILE} {DISAMBIGUATION_WORKING_DIR} {output}"
        )


rule generate_validation_result:
    input:
        DISAMBIGUATED_AUTHOR_LIST,
        GROUND_TRUTH_AUTHOR_LIST,
    output:
        DISAMBIGUATION_FOR_VALIDATION_TEST,
    run:
        shell("python workflow/generate-validation-file.py {input} {output}")


rule calc_f1score:
    input:
        DISAMBIGUATION_FOR_VALIDATION_TEST,
    output:
        VALIDATION_RESULT,
    run:
        shell("python workflow/calculate_pairwise_F1.py {input} {output}")


rule to_json2sql:
    input:
        WOS_JSON_DIR,
    output:
        WOS_SQL_TABLE,
    run:
        shell("python workflow/wosJson2Sql.py {input} {output}")


rule make_json_for_sampled_papers:
    input:
        WOS_JSON_DB,
        WOS_JSON_POS,
        WOS_UID_FILE_SAMPLED,
    output:
        WOS_JSON_FILE_SAMPLED,
    run:
        shell("python workflow/retrieve-json-file.py {input} {output}")
