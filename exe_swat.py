from models.sadi_wrapper import SADI_SWaT
from utils.utils import train, evaluate_anomalies
import numpy as np
import torch
import sys
import os
# from pypots.imputation import SAITS, BRITS
# import matplotlib.pyplot as plt
import matplotlib
# import pickle
from datasets.dataset_swat import get_dataloader, get_testloader_swat
# import json
from json import JSONEncoder
# import math
from config_ablation import common_config
# from sklearn.impute import KNNImputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

data = 'temps'
n_steps = 60 #366
n_features = 51
miss_type = 'random'
seed = np.random.randint(10,100)
dataset_name = 'nasce'
mean_std_file = f'./data/swat/X'
nsample = 50

 #352 #len(given_features)
train_loader, valid_loader = get_dataloader(mean_std_file, n_features=1, batch_size=16, missing_ratio=0.2)

model_folder = f"./saved_model_swat"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

config_dict_sadi = {
    'train': {
        'epochs': 5000, # 3000 -> ds3
        'batch_size': 16,
        'lr': 1.0e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end':  0.5,
        'num_steps': 50,
        'schedule': "cosine",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix", # noise mix
        'type': 'SAITS',
        'n_layers': 4,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': n_steps,
        'n_feature': n_features, #len(given_features),
        'd_model':  256,  # temps=1024,
        'd_inner': 128, # temps=512,
        'n_head': 8,
        'd_k': 64, #len(given_features),
        'd_v': 64, #len(given_features),
        'dropout': 0.1,
        'diagonal_attention_mask': False,
    },
    'ablation': {
        'fde-choice': 'fde-conv-multi',
        'fde-layers': 4,
        'is_fde': True,
        'weight_combine': True,
        'no-mask': False,
        'fde-diagonal': True,
        'is_fde_2nd': False,
        'reduce-type': 'linear',
        'is_2nd_block': True
    }
}
print(f"config: {config_dict_sadi}")
# name = 'fde-conv-multi'
config_dict_sadi['ablation'] = common_config['ablation']
config_dict_sadi['model']['n_layers'] = 4 #common_config['n_layers']
config_dict_sadi['name'] = common_config['name']
config_dict_sadi['ablation']['fde-layers'] = 4
config_dict_sadi['ablation']['is_2nd_block'] = True
config_dict_sadi['ablation']['is_fde'] = True
config_dict_sadi['ablation']['weight_combine'] = True # True
config_dict_sadi['ablation']['fde-time-pos-enc'] = False 
config_dict_sadi['name'] = f'swat'
name = config_dict_sadi['name']
print(config_dict_sadi)
model_sadi = SADI_SWaT(config_dict_sadi, device, target_dim=n_features).to(device)

filename = f"model_sadi_{name}.pth"
print(f"\n\DiffSAITS training starts.....\n")

model_sadi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# #
# train(
#     model_sadi,
#     config_dict_sadi["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{filename}",
#     is_saits=True
# )

data_folder = './results/swat/'

test_loader_1, test_loader_2 = get_testloader_swat(mean_std_file, n_features=1, batch_size=16, missing_ratio=0.5)

evaluate_anomalies(
    model_sadi,
    data_folder,
    test_loader_1=test_loader_1,
    test_loader_2=test_loader_2,
    test_labels_file=f'./data/swat/SWaT_minute_segments_anomaly_labels.npy'
)


