import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
import pandas as pd

from scipy.stats import spearmanr

in_data = pd.read_csv(sys.argv[1])

plt.rcParams.update({'font.size': 20})
plt.subplot(2,1,1)
#plt.scatter(in_data["correctness"],in_data["ef_mse"],marker=".")
plt.scatter(in_data["sg_mse"],in_data["correctness"],marker="x",alpha=0.5,color="blue")
plt.plot(np.unique(in_data["sg_mse"]), np.poly1d(np.polyfit(in_data["sg_mse"], in_data["correctness"], 1))(np.unique(in_data["sg_mse"])),color="green")

plt.grid()
plt.ylabel(r"Correctness $i$")
plt.xlabel(r"Spec MSE")
print("spearmanr sg_mse",spearmanr(in_data["correctness"],in_data["sg_mse"])[0])
print("pearsonr sg_mse",np.corrcoef(in_data["correctness"],in_data["sg_mse"])[0,1])
print("RMSE sg_mse",np.sqrt(np.mean(np.square(in_data["correctness"]-in_data["sg_mse"]))))


plt.savefig("spec.png")
plt.savefig("spec.svg")