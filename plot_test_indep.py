import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys
from scipy.optimize import curve_fit


gt_data= pd.read_json(sys.argv[1])
pred_data = pd.read_csv(sys.argv[2])

gt_data["predicted"] = pred_data["predicted"]



def logit_func(x,a,b):
    return 1/(1+np.exp(a*x+b))



    
y = gt_data["correctness"].values/100.00
x = gt_data["predicted"].values/100.00


# logistic mapping curve fit

popt,_ = curve_fit(logit_func, x, y)
a,b = popt
#a,b =(-5.147223278893597,2.944053164147111)
#make the scatter plot
plt.scatter(x,y,marker='.',alpha=0.5)

plt.plot(np.linspace(0,1,len(x)),logit_func(np.linspace(0,1,len(x)),a,b),color="red")
plt.grid()


# decorate the plot
plt.xlim([-.1,1.1])
plt.ylim([-.1,1.1])
plt.ylabel(r"Ground Truth Correctness $i$")
plt.xlabel(r"Predicted Correctness $\hat{i}$")
plt.title(f"{sys.argv[2].split('_')[-1].strip('.csv')} Predicted vs Ground Truth Inteligibility")
plt.savefig(f"{sys.argv[2].split('/')[0]}/{sys.argv[2].split('_')[-2].strip('.csv')}_plot.png")

