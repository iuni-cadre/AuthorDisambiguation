#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from collections import Counter
import sys


def calculate_pairwise_correct_and_false_pair(author_group_dict):
    TP = 0
    FP = 0
    for id_, row in author_group_dict.items():
        length = len(row)
        counter_dict = Counter(row)
        temp_TP = 0
        for k, v in counter_dict.items():
            if v > 1:
                temp_TP += v * (v - 1) / 2
        temp_FP = length * (length - 1) / 2 - temp_TP

        TP += temp_TP
        FP += temp_FP

    return TP, FP


def calculate_pairwise_F_measure(
    true_author_ids_group_by_predicted_result, predicted_result_group_by_true_author
):
    TP, FP = calculate_pairwise_correct_and_false_pair(
        true_author_ids_group_by_predicted_result
    )
    _, FN = calculate_pairwise_correct_and_false_pair(
        predicted_result_group_by_true_author
    )

    pairwise_precision = TP / (TP + FP)
    pairwise_recall = TP / (TP + FN)
    pairwise_F = (2 * pairwise_precision * pairwise_recall) / (
        pairwise_precision + pairwise_recall
    )

    return pairwise_F


if __name__ == "__main__":

    INPUT_FILE = sys.argv[
        1
    ]  # "/gpfs/sciencegenome/WoS-disambiguation/validation/validation-disambiguated-authors.csv"
    OUTPUT_FILE = sys.argv[2]
    FILE = open(OUTPUT_FILE, "w")

    #
    # Data
    #
    data = pd.read_csv(INPUT_FILE)

    #
    # Process
    #

    # Grouping
    true_author_ids_group_by_predicted_result = {
        id_: list(rows["true_author_id"])
        for id_, rows in data.groupby("predicted_author_id")
    }
    predicted_result_group_by_true_author = {
        id_: list(rows["predicted_author_id"])
        for id_, rows in data.groupby("true_author_id")
    }

    # Calculate the F1-score
    score = calculate_pairwise_F_measure(
        true_author_ids_group_by_predicted_result, predicted_result_group_by_true_author
    )

    # For validation (compute with the perfect case)
    # perfect_true_author_ids_group_by_predicted_result = {
    #    id_: list(rows["predicted_author_id"])
    #    for id_, rows in data.groupby("predicted_author_id")
    # }
    # perfect_predicted_result_group_by_true_author = {
    #    id_: list(rows["true_author_id"]) for id_, rows in data.groupby("true_author_id")
    # }
    #
    #
    #
    # calculate_pairwise_F_measure(
    #    perfect_true_author_ids_group_by_predicted_result,
    #    perfect_predicted_result_group_by_true_author,
    # )
    FILE.write("F1-score:{score}".format(score=score))
    FILE.close()  # to change file access modes
