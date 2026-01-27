import numpy as np
import json
from json import JSONDecoder
import os
import matplotlib.pyplot as plt

class NumpyArrayDecoder(JSONDecoder):
    def default(self, obj):
        if isinstance(obj, list):
            return np.array(obj)
        return JSONDecoder.default(self, obj)
    
anomaly_results_folder = './results/swat/'
anomaly_results_file = f"{anomaly_results_folder}/anomaly_results.json"

results_mse = []
with open(anomaly_results_file, 'r') as f:
    anomaly_results = json.load(f, cls=NumpyArrayDecoder)
length = len(anomaly_results.keys())
for i in range(1, length+1):
    result = anomaly_results[str(i)]
    results_mse.append(result['mse'])
results_mse = np.concatenate(results_mse, axis=0)

label_data_file = f'./data/swat/SWaT_minute_segments_anomaly_labels.npy'
labels = np.load(label_data_file)

print(f"mse: {results_mse.shape}, labels: {labels.shape}")

for sample_idx in range(results_mse.shape[0]):  # Change range for different samples
    mse_seq = results_mse[sample_idx]
    label_seq = labels[sample_idx]
    x = np.arange(len(mse_seq))

    plt.figure(figsize=(12, 4))
    plt.plot(x, mse_seq, label="MSE", linewidth=2)

    # Highlight anomaly regions
    in_anomaly = False
    start = 0

    for i in range(len(label_seq)):
        if label_seq[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif label_seq[i] == 0 and in_anomaly:
            plt.axvspan(start, i, alpha=0.3, label="Anomaly" if start == 0 else "")
            in_anomaly = False

    # If sequence ends during anomaly
    if in_anomaly:
        plt.axvspan(start, len(label_seq), alpha=0.3, label="Anomaly")

    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.title(f"MSE with Anomaly Regions (Sample {sample_idx})")
    plt.legend()
    plt.tight_layout()
    if not os.path.isdir(f"{anomaly_results_folder}/plots"):
        os.makedirs(f"{anomaly_results_folder}/plots")
    plt.savefig(f"{anomaly_results_folder}/plots/mse_anomalies_sample_{sample_idx}.png")


