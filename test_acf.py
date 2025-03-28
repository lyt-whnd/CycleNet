import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np

from scipy.signal import find_peaks

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

df_raw = pd.read_csv('dataset/ETTh1.csv')
scaler = StandardScaler()
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]

train_data = df_data[:12*30*24]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)

train_data = data[:12*30*24]
test_data = data[16*30*24:20*30*24]


acf_values = acf(train_data[:,0], nlags=200)

peaks, _ = find_peaks(acf_values)
# If the first detected peak is at lag 0 (trivial), choose the next one as the second peak
if len(peaks) > 0:
    if peaks[0] == 0 and len(peaks) > 1:
        second_peak = peaks[1]
    else:
        second_peak = peaks[0]
else:
    second_peak = None

lags = np.arange(len(acf_values))
plt.figure(figsize=(3, 2.5))
plt.bar(lags, acf_values)

plt.xlim([0, 200])
plt.ylim([0, 1])

plt.xlabel("Lags", fontsize=10)
plt.ylabel("Autocorrelation", fontsize=10)

plt.grid(True)
plt.savefig('acf/etth1[0].pdf', bbox_inches='tight')
plt.show()

if second_peak is not None:
    print(f"The second peak is at lag {second_peak} with an ACF value of {acf_values[second_peak]:.4f}")
else:
    print("No second peak found in the ACF values.")