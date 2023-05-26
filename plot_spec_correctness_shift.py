import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
import pandas as pd

from scipy.stats import spearmanr
from scipy.optimize import curve_fit
def logit(x,a,b):
    return 1/(1+np.exp(-a*(x-b)))
in_data_HL = pd.read_csv(sys.argv[1])
in_data_no_HL = pd.read_csv(sys.argv[2])



plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,7.5))
plt.suptitle("Correctness VS MSE for Spectrogram")
plt.subplot(2,1,1)
plt.scatter(in_data_HL["sg_mse"],in_data_HL["correctness"],marker=".",alpha=0.5,color="blue",label=r"$\mathrm{SPEC}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.scatter(in_data_no_HL["sg_mse"],in_data_no_HL["correctness"],marker="x",alpha=0.5,color="red",label=r"$\mathrm{SPEC}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")
plt.xlabel(r"$\mathrm{SPEC}$ MSE")
plt.ylabel(r"Correctness $i$")
plt.legend(fontsize="small",loc="lower right")

plt.subplot(2,1,2)
plt.scatter(in_data_no_HL["sg_mse"],in_data_HL["sg_mse"],marker=".",alpha=0.5,color="green",label="HL")
plt.ylabel(r"$\mathrm{SPEC}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.xlabel(r"$\mathrm{SPEC}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")

plt.grid()
plt.tight_layout()
plt.savefig(sys.argv[1].split(".")[0]+"_spec_comparrision.png")
plt.savefig(sys.argv[1].split(".")[0]+"_spec_comparrison.svg")