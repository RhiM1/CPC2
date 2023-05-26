import numpy as np
from scipy.optimize import curve_fit
import scipy as sy
import matplotlib.pyplot as plt

d = np.array([75, 80, 90, 95, 100, 105, 110, 115, 120, 125], dtype=float)/125
p2 = np.array([6, 13, 25, 29, 29, 29, 30, 29, 30, 30], dtype=float) / 30. # scale to 0..1

# psychometric function
def pf(x, alpha, beta):
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

def logit_func(x,a,b):
    return 1/(1+np.exp(a*x+b))
# fitting
par0 = sy.array([0., 1.]) # use some good starting values, reasonable default is [0., 1.]
par, mcov = curve_fit(logit_func, d, p2, par0)
print(par)
plt.plot(d, p2, 'ro')
plt.plot(d, logit_func(d, par[0], par[1]))
plt.savefig("test2.png")
