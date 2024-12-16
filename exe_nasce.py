from models.sadi_wrapper import SADI_NASCE
from utils.utils import train, get_num_params, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS, BRITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
from datasets.dataset_nasce import get_dataloader
import json
from json import JSONEncoder
import math
from config_ablation import common_config, partial_bm_config
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
n_steps = 366 #366
n_features = 352 if data == 'temps' else 775
miss_type = 'random'
seed = np.random.randint(10,100)
dataset_name = 'nasce'
data_file_train = f'./data/nasce/X_OR_{data}_train.npy'
mean_std_file = f'./data/nasce/X_OR_{data}'
data_file_test = f'./data/nasce/X_OR_{data}_test.npy'
nsample = 50

 #352 #len(given_features)
train_loader, valid_loader = get_dataloader(data_file_train, mean_std_file, batch_size=2, missing_ratio=0.2, type=data_type, data=data)

model_folder = f"./saved_model_{dataset_name}"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

config_dict_sadi = {
    'train': {
        'epochs': 15000, # 3000 -> ds3
        'batch_size': 16,
        'lr': 1.0e-4 if data == 'ppt' else 1e-4# temps = 1e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end':  0.5 if data == 'ppt' else 0.2, # temps: 0.3,
        'num_steps': 50,
        'schedule': "cosine",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix" if data == 'ppt' else "random", # noise mix
        'type': 'SAITS',
        'n_layers': 4,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': n_steps,
        'n_feature': n_features, #len(given_features),
        'd_model':  1300 if data == 'ppt' else 1024,  # temps=1024,
        'd_inner': 768 if data == 'ppt' else 512, # temps=512,
        'n_head': 8,
        'd_k': 128 if data == 'ppt' else 64, #len(given_features),
        'd_v': 128 if data == 'ppt' else 64, #len(given_features),
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
config_dict_sadi['ablation']['is_fde'] = True if data == 'ppt' else True
config_dict_sadi['ablation']['is-fde-linear'] = False if data == 'ppt' else False
config_dict_sadi['ablation']['weight_combine'] = True # True
config_dict_sadi['ablation']['fde-time-pos-enc'] = False if data == 'ppt' else False
config_dict_sadi['name'] = f'skip_no_fde_1st_mask_pos_enc_loss_p_{data}_all' if data == 'ppt' else f'meeting'
name = config_dict_sadi['name']
print(config_dict_sadi)
model_sadi = SADI_NASCE(config_dict_sadi, device, target_dim=n_features).to(device)

filename = f"model_diffsaits_{name}.pth"
print(f"\n\DiffSAITS training starts.....\n")

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# #
train(
    model_sadi,
    config_dict_sadi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True
)


config_dict_sadi_pbm = {
    'train': {
        'epochs': 10000, # 3000 -> ds3
        'batch_size': 16,
        'lr': 1.0e-4 if data == 'ppt' else 1e-4# temps = 1e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end':  0.5 if data == 'ppt' else 0.2, # temps: 0.3,
        'num_steps': 50,
        'schedule': "cosine",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix" if data == 'ppt' else "pbm", # noise mix
        'type': 'SAITS',
        'n_layers': 4,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': n_steps,
        'n_feature': n_features, #len(given_features),
        'd_model':  1300 if data == 'ppt' else 1024,  # temps=1024,
        'd_inner': 768 if data == 'ppt' else 512, # temps=512,
        'n_head': 8,
        'd_k': 128 if data == 'ppt' else 64, #len(given_features),
        'd_v': 128 if data == 'ppt' else 64, #len(given_features),
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
print(f"config: {config_dict_sadi_pbm}")
# name = 'fde-conv-multi'
config_dict_sadi_pbm['ablation'] = common_config['ablation']
config_dict_sadi_pbm['model']['n_layers'] = 4 #common_config['n_layers']
config_dict_sadi_pbm['name'] = common_config['name']
config_dict_sadi_pbm['ablation']['fde-layers'] = 4
config_dict_sadi_pbm['ablation']['is_2nd_block'] = False
config_dict_sadi_pbm['ablation']['is_fde'] = True if data == 'ppt' else True
config_dict_sadi_pbm['ablation']['is-fde-linear'] = False if data == 'ppt' else False
config_dict_sadi_pbm['ablation']['weight_combine'] = True
config_dict_sadi_pbm['ablation']['fde-time-pos-enc'] = False if data == 'ppt' else False
config_dict_sadi_pbm['name'] = f'nacse_temp'
name = config_dict_sadi_pbm['name']
print(config_dict_sadi_pbm)
model_sadi_pbm = SADI_NASCE(config_dict_sadi_pbm, device, target_dim=n_features).to(device)

# model_sadi_pbm.load_state_dict(torch.load(f"{model_folder}/{filename}"))
filename = f"model_sadi_pbm_{name}.pth"
print(f"\n\nSADI PBM training starts.....\n")

# 
#
train(
    model_sadi_pbm,
    config_dict_sadi_pbm["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True,
    pbm_start=0.001
)

# model_sadi_pbm.load_state_dict(torch.load(f"{model_folder}/{filename}"))


models = {
    # 'CSDI': model_csdi,
    # 'SAITS': saits,
    'SADI': model_sadi,
    'SADI_pbm': model_sadi_pbm,
    # 'BRITS': brits_model,
    # 'MICE': mice,
    # 'KNN': knn
}
# mse_folder = f"results_{dataset_name}_{name}_new/metric"
# data_folder = f"results_{dataset_name}_{name}_new/data"
# name = miss_type
partial_bm_config['length_range'] = (30, 30)
partial_bm_config['features'] = "all"
mse_folder = f"results_nasce_{name}_{data}/metric"
data_folder = f"results_nasce_{name}_{data}/data"



filename = (data_file_test, mean_std_file)
pbm = [2,10,50,90,100]
for bm in pbm:  
    partial_bm_config['features'] = bm
    print(f"partial bm: {bm}")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='nasce', batch_size=16, partial_bm_config=partial_bm_config, filename=filename, data=False, unnormalize=False)

# lengths = [50, 100, 150]
# for l in lengths:
#     print(f"\nBlackout length = {l}")
#     evaluate_imputation_all(models=models, trials=5, mse_folder=mse_folder, dataset_name='nasce', batch_size=16, length=l, filename=filename)
# #     # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='nasce', length=l, trials=1, batch_size=1, data=True, filename=filename)

# lengths = [50, 100, 150]
# for l in lengths:
#     print(f"\nForecasting = {l}")
#     evaluate_imputation_all(models=models, trials=1, mse_folder=mse_folder, dataset_name='nasce', batch_size=16, length=l, forecasting=True, filename=filename)
# #     # evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='nasce', length=l, trials=1, batch_size=1, data=True, filename=filename)

# miss_ratios = [0.2, 0.5, 0.8]
# for ratio in miss_ratios:
#     print(f"\nRandom Missing: ratio ({ratio})")
#     evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='nasce', batch_size=16, missing_ratio=ratio, random_trial=True, filename=filename, rmse=True)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='nasce', trials=1, batch_size=1, data=True, missing_ratio=ratio, random_trial=True, filename=filename)
