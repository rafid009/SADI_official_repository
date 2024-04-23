from models.main_model import SADI_Synth
from datasets.dataset_synth import get_dataloader, get_testloader
from utils.utils import train, get_num_params, calc_quantile_CRPS, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS, BRITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
from datasets.synthetic_data import create_synthetic_data_v1, feats_v5
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

given_features = feats_v5

seed = np.random.randint(10, 100)
nsample = 50

n_steps = 100
n_features = len(given_features)
num_seasons = 50
noise = False
train_loader, valid_loader, mean, std = get_dataloader(n_steps, n_features, num_seasons, batch_size=32, missing_ratio=0.1, seed=seed, is_test=False, v2='v5', noise=noise)

config_sadi = {
    'train': {
        'epochs':9000, # 3000 -> ds3
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
        'schedule': "quad",
         'is_fast': False,
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix", # noise mix
        'type': 'SAITS',
        'n_layers': 3,
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
config_sadi['model']['n_layers'] = common_config['n_layers']
config_sadi['name'] = common_config['name']
print(f"config: {config_sadi}")
name = config_sadi['name']
model_sadi = SADI_Synth(config_sadi, device, target_dim=len(given_features)).to(device)

filename = f"model_SADI_synth_v5_{name}_new.pth"
print(f"\n\SADI training starts.....\n")
model_folder = "saved_model_v5"

train(
    model_sadi,
    config_sadi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}"
)

model_sadi.load_state_dict(torch.load(f"{model_folder}/{filename}"))


models = {
    'SADI': model_sadi
}
mse_folder = f"results_synth_v5_{name}_{partial_bm_config['features']}/metric"
data_folder = f"results_synth_v5_{name}_{partial_bm_config['features']}/data"

pbm = [1,2,3,4,5]
for bm in pbm:
    partial_bm_config['features'] = bm
    print(f"partial features: {partial_bm_config['features']}")
    evaluate_imputation_all(models=models, trials=1, mse_folder=mse_folder, dataset_name='synth_v5', batch_size=32, mean=mean, std=std, partial_bm_config=partial_bm_config)


lengths = [10, 50, 90]
for l in lengths:
    print(f"\nlength = {l}")
    print(f"\nBlackout:")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v5', batch_size=32, length=l, noise=noise, mean=mean, std=std)

print(f"\nForecasting:")
evaluate_imputation_all(models=models, trials=1, mse_folder=mse_folder, dataset_name='synth_v5', batch_size=32, length=(10, 80), forecasting=True, noise=noise, mean=mean, std=std)

miss_ratios = [0.1, 0.5, 0.9]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='synth_v5', batch_size=32, missing_ratio=ratio, random_trial=True, noise=noise, mean=mean, std=std)
