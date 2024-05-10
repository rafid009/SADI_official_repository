from models.sadi_wrapper import SADI_AWN
from datasets.dataset_awn import get_dataloader, get_testloader_AWN
from utils.utils import train, get_num_params, calc_quantile_CRPS, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS, BRITS
import matplotlib.pyplot as plt
import matplotlib
import pickle

import json
from json import JSONEncoder
import math
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config_ablation import common_config, partial_bm_config
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# torch.manual_seed(42)
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=torch.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

given_features = [
    # 'TSTAMP_PST', 
    # ' UNIT_ID', 
    # ' STATION_NAME', 
    # ' LATITUDE', 
    # ' LONGITUDE', 
    # ' ELEVATION_FEET', 
    'AIR_TEMP_F',
    'SECOND_AIR_TEMP_F',
    'AIR_TEMP_9M_F',
    'RELATIVE_HUMIDITY_%',
    'DEWPOINT_F',
    'LEAF_WETNESS',
    'PRECIP_INCHES',
    'SECOND_PRECIP_INCHES',
    'WIND_DIRECTION_2M_DEG',
    'WIND_SPEED_2M_MPH',
    'WIND_GUST_2M_MPH',
    'WIND_DIRECTION_10M_DEG',
    'WIND_SPEED_10M_MPH',
    'WIND_GUST_10M_MPH',
    'SOLAR_RAD_WM2',
    'SOIL_TEMP_2_IN_F',
    'SOIL_TEMP_8_IN_F',
    'SOIL_TEMP_24_IN_F',
    'SOIL_MOIS_8_IN_%',
    'SOIL_WP_2_IN_KPA',
    'SOIL_WP_8_IN_KPA',
    'SOIL_WP_24_IN_KPA',
    'MSLP_HPA' 
]

seed = np.random.randint(10, 100)
nsample = 50

n_steps = 672
n_features = len(given_features)
# num_seasons = 50
noise = False
filename = '330141.csv'

train_loader, valid_loader = get_dataloader(n_steps, (filename, filename), batch_size=8, missing_ratio=0.2, seed=seed)




config_sadi = {
    'train': {
        'epochs':1900, # 3000 -> ds3
        'batch_size': 16 ,
        'lr': 1.0e-4
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 50,
        'schedule': "cosine",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random", # noise mix
        'type': 'SAITS',
        'n_layers': 4,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': n_steps,
        'n_feature': len(given_features),
        'd_model': 128, # 256 for enc-dec
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64, #len(given_features),
        'd_v': 64, #len(given_features),
        'dropout': 0.1,
        'diagonal_attention_mask': False
    },
    'ablation': {
        'fde-choice': 'fde-conv-multi',
        'fde-layers': 3,
        'is_fde': True,
        'weight_combine': True,
        'fde-no-mask': True,
        'fde-diagonal': False,
        'is_fde_2nd': False,
        'fde-pos-enc': False,
        'reduce-type': 'linear',
        'embed-type': 'linear',
        'is_2nd_block': True,
        'is-not-residual': False,
        'res-block-mask': False,
        'is-fde-loop': False,
        'skip-connect-no-res-layer': False,
        'enc-dec': False,
        'is_stable': True,
        'is_first': False,
        'blackout': False,
        'is_dual': False
    }
}
config_sadi['ablation'] = common_config['ablation']

config_sadi['name'] = common_config['name']
print(f"config: {config_sadi}")
name = config_sadi['name']
model_sadi = SADI_AWN(config_sadi, device, target_dim=len(given_features)).to(device)

model_filename = f"model_SADI_awn_{filename.split('.')[0]}.pth"
print(f"\n\SADI training starts.....\n")
model_folder = "saved_model_awn"

# train(
#     model_sadi,
#     config_sadi["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=f"{model_filename}",
# )

model_sadi.load_state_dict(torch.load(f"{model_folder}/{model_filename}"))


models = {
    'SADI': model_sadi
}
mse_folder = f"results_awn_{name}_{partial_bm_config['features']}/metric"
data_folder = f"results_awn_{name}_{partial_bm_config['features']}/data"


pbm = [11]
for bm in pbm:  
    partial_bm_config['features'] = bm
    print(f"features: {partial_bm_config['features']}")
    # evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, test_indices=[32,33], mean=mean, std=std, partial_bm_config=partial_bm_config)
    evaluate_imputation_all(models=models, filename=filename, trials=1, mse_folder=data_folder, dataset_name='awn', batch_size=1, partial_bm_config=partial_bm_config, data=True, unnormalize=True)


lengths = [20, 50, 80]
for l in lengths:
    print(f"\nlength = {l}")
    print(f"\nBlackout:")
    evaluate_imputation_all(models=models, filename=filename, trials=10, mse_folder=mse_folder, dataset_name='awn', batch_size=8, length=l, unnormalize=True)

# print(f"\nForecasting:")
# evaluate_imputation_all(models=models, trials=1, mse_folder=mse_folder, dataset_name='synth_v1', batch_size=32, length=(10, 80), forecasting=True, noise=noise, mean=mean, std=std)

miss_ratios = [0.2, 0.5, 0.8]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})")
    evaluate_imputation_all(models=models, filename=filename, trials=10, mse_folder=mse_folder, dataset_name='awn', batch_size=8, missing_ratio=ratio, random_trial=True, unnormalize=True)
