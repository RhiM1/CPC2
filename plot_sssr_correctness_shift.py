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

if "hubert" in sys.argv[1]:
    in_data_no_HL = in_data_no_HL[in_data_HL["ef_mse"]<0.0005]
    in_data_HL = in_data_HL[in_data_HL["ef_mse"]<0.0005]

#in_data_HL["ef_mse"] = in_data_HL["ef_mse"]/np.max(in_data_HL["ef_mse"])
#in_data_HL["ol_mse"] = in_data_HL["ol_mse"]/np.max(in_data_HL["ol_mse"])
#in_data_no_HL["ef_mse"] = in_data_no_HL["ef_mse"]/np.max(in_data_no_HL["ef_mse"])
#in_data_no_HL["ol_mse"] = in_data_no_HL["ol_mse"]/np.max(in_data_no_HL["ol_mse"])

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(15,10))
plt.suptitle("Correctness VS MSE for %s"%sys.argv[1].split(".")[0].split("_")[-1].upper())
plt.subplot(2,2,1)
plt.scatter(in_data_HL["ef_mse"],in_data_HL["correctness"],marker=".",alpha=0.5,color="blue",label=r"$\mathrm{FE}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.scatter(in_data_no_HL["ef_mse"],in_data_no_HL["correctness"],marker="x",alpha=0.5,color="red",label=r"$\mathrm{FE}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")
plt.xlabel(r"$\mathcal{G}_\mathrm{FE}$ MSE")
plt.ylabel(r"Correctness $i$")
plt.legend(loc="lower right",fontsize="small")

plt.subplot(2,2,2)
plt.scatter(in_data_HL["ol_mse"],in_data_HL["correctness"],marker=".",alpha=0.5,color="blue",label=r"$\mathrm{OL}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.scatter(in_data_no_HL["ol_mse"],in_data_no_HL["correctness"],marker="x",alpha=0.5,color="red",label=r"$\mathrm{OL}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")
plt.xlabel(r"$\mathcal{G}_\mathrm{OL}$ MSE")
plt.ylabel(r"Correctness $i$")
plt.legend(loc="lower right",fontsize="small")

plt.subplot(2,2,3)
plt.scatter(in_data_no_HL["ef_mse"],in_data_HL["ef_mse"],marker=".",alpha=0.5,color="green",label=r"$\mathbf{\hat{s}[n]'}$")
plt.ylabel(r"$\mathrm{FE}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.xlabel(r"$\mathrm{FE}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")
plt.grid()

plt.subplot(2,2,4)
plt.scatter(in_data_no_HL["ol_mse"],in_data_HL["ol_mse"],marker="x",alpha=0.5,color="green",label=r"$\mathbf{\hat{s}[n]'}$")
plt.ylabel(r"$\mathrm{OL}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]'})$")
plt.xlabel(r"$\mathrm{OL}_\mathrm{MSE}(\mathbf{s[n]},\mathbf{\hat{s}[n]})$")
plt.grid()
plt.tight_layout()
plt.savefig(sys.argv[1].split(".")[0]+"_comparrision.png")
plt.savefig(sys.argv[1].split(".")[0]+"_comparrison.svg")
