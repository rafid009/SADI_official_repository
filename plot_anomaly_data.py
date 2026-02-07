import os
import matplotlib.pyplot as plt
import numpy as np

input_folder = './data/swat'
data = np.load(f"{input_folder}/SWaT_minute_segments_anomaly.npy")
features = ['FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','FIT601','P601','P602','P603']
out_folder = 'anomaly_plots'
i = 0
x = np.arange(ts.shape[0])
for ts in data:
    for j in range(ts.shape[1]):
        y = ts[:,j]
        plt.figure(figsize=(12, 4))
        plt.plot(x, y, label=features[j], linewidth=2)
        plt.xlabel("Time Step (s)")
        plt.ylabel(features[j])
        plt.title(f"Time-series with Anomaly Regions for feature = {features[j]} (Sample {i})")
        plt.legend()
        plt.tight_layout()

        folder = f"{out_folder}/{i}"
        if not os.path.isdir(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}/{features[j]}.png")
