"""
Score a set of intelligibility predictions

Usage:
  python3 compute_rmse_score.py <CPC1_METADATA_JSON_FILE> <PREDICTION_CSV_FILE>

e.g.
  python3 compute_rmse_score.py "$CLARITY_DATA"/metadata/CPC1.train.json predictions.csv

Requires the predictions.csv file containing the predicted intelligibility
for each signal. This should have the following format.

  scene,listener,system,predicted
  S09637,L0240,E013,81.72
  S08636,L0218,E013,90.99
  S08575,L0236,E013,81.43
etc
"""

import argparse

import numpy as np
import pandas as pd
import csv
from scipy.stats import spearmanr


DATAROOT = "/store/store1/data/clarity_CPC2_data/" 
DATAROOT_CPC1 = "/store/store1/data/clarity_CPC1_data/" 
# DATAROOT = "~/exp/data/clarity_CPC1_eval_data/" 


def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))
def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))

def main(args):
    """Compute and report the RMS error comparing true intelligibilities
    with the predicted intelligibilities

    The true intelligibilities are stored in the CPC1 JSON metadata file
    that is supplied with the challenge. The predicted scores should be
    stored in a csv file with a format as follows

    scene,listener,system,predicted
    S09637,L0240,E013,81.72
    S08636,L0218,E013,90.99
    S08575,L0236,E013,81.43

    For example, as produced by the baseline system script predict_intel.py
    ...

    Args:
        intel_file_json (str): Name of the CPC JSON metadata storing true scores
        prediction_file_csv (str): Name of the csv file scoring predicted scores
    """

    # Load the predictions and the actual intelligibility data
    df_predictions = pd.read_csv(args.predictions_csv_file)
    df_intel = pd.read_json(args.test_json_file)

    # Merge into a common dataframe
    data = pd.merge(
        df_predictions,
        df_intel[["scene", "listener", "system", "correctness"]],
        how="left",
        on=["scene", "listener", "system"],
    )

    # remove the rows fore listener L0227
    #data = data[data["listener"] != "L0227"]


    # Compute the score comparing predictions with the actual
    # word correctnesses recorded by the listners
    if args.use_fitted:
        error = rmse_score(data["predicted_fitted"], data["correctness"])
        p_corr = np.corrcoef(data["predicted_fitted"],data["correctness"])[0][1]
        s_corr = spearmanr(data["predicted_fitted"],data["correctness"])[0]
        std = std_err(data["predicted_fitted"], data["correctness"])

    else:
        error = rmse_score(data["predicted"], data["correctness"])
        p_corr = np.corrcoef(data["predicted"],data["correctness"])[0][1]
        s_corr = spearmanr(data["predicted"],data["correctness"])[0]
        std = std_err(data["predicted"], data["correctness"])
    #print(p_corr)
    print(f"{args.test_json_file}: RMS prediction error: {error:5.2f} +/- {std:5.2f}")

    #print(f"RMS prediction error: {error:5.2f}")
    print(f"Pearson correlation: {p_corr:5.2f}")
    print(f"Spearman correlation: {s_corr:5.2f}")
    with open (args.output_file, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([args.predictions_csv_file.split("/")[-1].strip(".csv"), error, std, p_corr, s_corr])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions_csv_file", help="csv file containing the predicted intelligibilities"
    )
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC1 evaluation metadata", default = None
    )
    parser.add_argument(
        "--output_file", help="csv file containing the predicted intelligibilities", default = None
    )
    parser.add_argument(
        "--use_CPC1", help="train and evaluate on CPC1 data" , default=False, action='store_true'
    )
    parser.add_argument(
        "--use_fitted", help="use fitted data" , default=False, action='store_true'
    )

    args = parser.parse_args()
    args.dataroot = DATAROOT_CPC1 if args.use_CPC1 else DATAROOT
    if args.test_json_file is None:
        if args.use_CPC1: 
            args.test_json_file = args.dataroot + "metadata/CPC1.test_indep.json"
    if args.output_file is None:
        args.output_file = "save/"
        if args.use_CPC1:
            args.output_file = args.output_file + "CPC1_out"
        else:
            args.output_file = args.output_file + "CPC2_out"
        if args.use_fitted:
            args.output_file = args.output_file + "_fitted"
        args.output_file = args.output_file + ".csv"

        # args.output_file = "_".join(args.predictions_csv_file.split("_")[0:-1]) + "_res.csv"
        print(args.output_file)

    main(args)
    # main(args.cpc1_test_json_file, args.predictions_csv_file)
