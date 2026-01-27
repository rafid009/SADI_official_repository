import numpy as np
import json
from json import JSONDecoder
import os

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
for i in range(length):
    result = anomaly_results[str(i)]
    results_mse.append(result['mse'])
results_mse = np.concatenate(results_mse, axis=0)

label_data_file = f'./data/swat/SWaT_minute_segments_anomaly_labels.npy'
labels = np.load(label_data_file)

print(f"mse: {results_mse.shape}, labels: {labels.shape}")


