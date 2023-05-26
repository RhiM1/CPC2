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
import matplotlib.pyplot as plt

def rmse_score(x, y):
    return np.sqrt(np.mean((x - y) ** 2))
def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))

def main(intel_file_json, prediction_file_csv):
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
    df_predictions = pd.read_csv(prediction_file_csv)
    df_intel = pd.read_json(intel_file_json)

    # Merge into a common dataframe
    data = pd.merge(
        df_predictions,
        df_intel[["scene", "listener", "system", "correctness"]],
        how="left",
        on=["scene", "listener", "system"],
    )

    # Compute the score comparing predictions with the actual
    # word correctnesses recorded by the listners
    error = rmse_score(data["predicted"], data["correctness"])
    p_corr = np.corrcoef(data["predicted"],data["correctness"])[0][1]
    s_corr = spearmanr(data["predicted"],data["correctness"])[0]
    std = std_err(data["predicted"], data["correctness"])
    #print(p_corr)
    print(f"{intel_file_json}: RMS prediction error: {error:5.2f} +/- {std:5.2f}")

    #print(f"RMS prediction error: {error:5.2f}")
    print(f"Pearson corralation: {p_corr:5.2f}")
    print(f"Spearman corralation: {s_corr:5.2f}")
    
    system_dict_predicted = {}
    system_dict_correctness = {}
    for system in data["system"].unique():
        average_correctness = data[data["system"]==system]["correctness"].mean()
        average_predicted = data[data["system"]==system]["predicted"].mean()
        print(f"{system}: {average_correctness:5.2f} {average_predicted:5.2f}")
        print(average_correctness-average_predicted)
        system_dict_predicted[system] = average_predicted
        system_dict_correctness[system] = average_correctness
    X_axis = np.arange(len(system_dict_predicted))
    w = 0.35
    plt.figure(figsize=(13,5))
    #set the font size
    plt.rcParams.update({'font.size': 18})
    plt.subplot(2,1,1)
    plt.bar(X_axis,system_dict_predicted.values(),width=w,color="red", label=r"$\hat{i}$",alpha=0.75,edgecolor="black",linewidth=1)
    plt.bar(X_axis+w,system_dict_correctness.values(),width=w,color="green", label=r"$i$",alpha=0.75,edgecolor="black",linewidth=1)
    plt.xticks(X_axis+w/2,system_dict_predicted.keys())
    plt.xticks(rotation=45, ha="right",rotation_mode="anchor",size="medium")
    plt.ylim(0,100)
    plt.xlabel("System")
    plt.ylabel("Intelligibility")
    #plt.legend()
    plt.subplot(2,1,2)
    listener_dict_predicted = {}
    listener_dict_correctness = {}

    for listener in data["listener"].unique()[:15]:
        average_correctness = data[data["listener"]==listener]["correctness"].mean()
        average_predicted = data[data["listener"]==listener]["predicted"].mean()
        print(f"{listener}: {average_correctness:5.2f} {average_predicted:5.2f}")
        listener_dict_predicted[listener] = average_predicted
        listener_dict_correctness[listener] = average_correctness
    X_axis = np.arange(len(listener_dict_predicted))
    plt.bar(X_axis,listener_dict_predicted.values(),width=w,color="red", label=r"$\hat{i}$",alpha=0.75,edgecolor="black",linewidth=1)
    plt.bar(X_axis+w,listener_dict_correctness.values(),width=w,color="green", label=r"$i$",alpha=0.75,edgecolor="black",linewidth=1)
    plt.xticks(X_axis+w/2,listener_dict_predicted.keys())
    plt.xticks(rotation=45, ha="right",rotation_mode="anchor",size="medium")
    plt.ylim(0,100)

    plt.xlabel("Listener")
    plt.ylabel("Intelligibility")
    #plt.legend()

    plt.tight_layout()

    plt.savefig("%s.png"%prediction_file_csv.replace(".csv","_detail_plots"))
    plt.savefig("%s.svg"%prediction_file_csv.replace(".csv","_detail_plots"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cpc1_train_json_file", help="JSON file containing the CPC1 training metadata"
    )
    parser.add_argument(
        "predictions_csv_file", help="csv file containing the predicted intelligibilities"
    )
  
    args = parser.parse_args()

    main(args.cpc1_train_json_file, args.predictions_csv_file)
