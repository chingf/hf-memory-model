import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open("grideval.p", "rb") as f:
    grideval = pickle.load(f)

df = pd.DataFrame(grideval)
df.to_csv("grideval.csv")

btsp = np.array(grideval["BTSP"])
nmean = np.array(grideval["NoiseMean"])
nstd = np.array(grideval["NoiseStd"])
jmean = np.array(grideval["JMean"])
jstd = np.array(grideval["JStd"])
btspscale = np.array(grideval["BTSPScale"])
scores = np.array(grideval["Score"])

results = {"Mean":[], "Std":[], "Score":[]}
for _nmean in [-0.2, -0.1, 0., 0.1, 0.2]:
    for _nstd in [0.1, 0.2, 0.3]:
        all_scores = scores[np.logical_and(nmean==_nmean, nstd==_nstd)]
        results["Mean"].append(_nmean)
        results["Std"].append(_nstd)
        results["Score"].append(all_scores.max())
results = pd.DataFrame(results)
results = results.pivot("Mean", "Std", "Score")
sns.heatmap(results)
plt.show()

results = {"Mean":[], "Std":[], "Score":[]}
for _jmean in [-0.2, -0.1, 0., 0.1, 0.2]:
    for _jstd in [0.1, 0.2, 0.3]:
        all_scores = scores[np.logical_and(jmean==_jmean, jstd==_jstd)]
        results["Mean"].append(_jmean)
        results["Std"].append(_jstd)
        results["Score"].append(all_scores.max())
results = pd.DataFrame(results)
results = results.pivot("Mean", "Std", "Score")
sns.heatmap(results)
plt.show()

import pdb; pdb.set_trace()
