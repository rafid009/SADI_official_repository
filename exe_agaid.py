from models.sadi_wrapper import SADI_Agaid
from datasets.dataset_agaid import get_dataloader
from utils.utils import train, get_num_params, evaluate_imputation_all
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS, BRITS
from datasets.process_data import *
import pickle
from config_ablation import common_config, partial_bm_config
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(10, 100)

miss_type = 'random'



# # dataset_name = 'agaid' + '_weather'
data_file = './data/agaid/ColdHardiness_Grape_Merlot_2.csv'
# config_dict_csdi = config_dict_csdi_pattern if miss_type.startswith('pattern') else config_dict_csdi_random

train_loader, valid_loader, mean, std = get_dataloader(
    seed=seed,
    filename=data_file,
    batch_size=16,
    missing_ratio=0.2,
    season_idx=[32, 33]
)


model_folder = "./saved_model_agaid"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)


    
config_dict_sadi = {
    'train': {
        'epochs': 9000,
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
        'is_fast': False
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix",
        'type': 'SAITS',
        'n_layers': 3,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': 252,
        'n_feature': len(features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}
config_dict_sadi['ablation'] = common_config['ablation']
config_dict_sadi['model']['n_layers'] = 3 # 3 #common_config['n_layers']
config_dict_sadi['ablation']['is_2nd_block'] = True
config_dict_sadi['ablation']['is_fde'] = True
config_dict_sadi['ablation']['fde-time-pos-enc'] = False
config_dict_sadi['ablation']['weight_combine'] = True #True
config_dict_sadi['ablation']['fde-layers'] = 3
config_dict_sadi['name'] = 'normal' # 'skip_fde_1st_mask_pos_enc_loss_p_bm' # just remove _new_loss
# config_dict_diffsaits['name'] = common_config['name']
name = config_dict_sadi['name'] #+ "_weather"
print(config_dict_sadi)
# model_diff_saits_simple = CSDI_Agaid(config_dict, device, is_simple=True).to(device)
model_sadi = SADI_Agaid(config_dict_sadi, device, is_simple=False).to(device)
filename = f'model_sadi_{name}.pth'

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# 
train(
    model_sadi,
    config_dict_sadi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True,
    data_type='agaid',
    pbm_start=-1
)
nsample = 50
# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
print(f"SADI params: {get_num_params(model_sadi)}")


config_dict_sadi_pbm = {
    'train': {
        'epochs': 2000,
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
        'is_fast': False
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "pbm",
        'type': 'SAITS',
        'n_layers': 3,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': 252,
        'n_feature': len(features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}
config_dict_sadi_pbm['ablation'] = common_config['ablation']
config_dict_sadi_pbm['model']['n_layers'] = 3 # 3 #common_config['n_layers']
config_dict_sadi_pbm['ablation']['is_2nd_block'] = True
config_dict_sadi_pbm['ablation']['is_fde'] = True
config_dict_sadi_pbm['ablation']['fde-time-pos-enc'] = False
config_dict_sadi_pbm['ablation']['weight_combine'] = True
config_dict_sadi_pbm['ablation']['fde-layers'] = 3
config_dict_sadi_pbm['name'] = 'normal' # "no_2nd_block" # 'skip_fde_1st_mask_pos_enc_loss_p_bm' # just remove _new_loss
# config_dict_diffsaits['name'] = common_config['name']
name = config_dict_sadi_pbm['name'] #+ "_weather"
print(config_dict_sadi_pbm)
# model_diff_saits_simple = CSDI_Agaid(config_dict, device, is_simple=True).to(device)
model_sadi_pbm = SADI_Agaid(config_dict_sadi_pbm, device, is_simple=False).to(device)


model_sadi_pbm.load_state_dict(torch.load(f"{model_folder}/{filename}"))
filename = f'model_sadi_pbm_{name}.pth'
train(
    model_sadi_pbm,
    config_dict_sadi_pbm["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=f"{filename}",
    is_saits=True,
    data_type='agaid',
    pbm_start=0.001
)
nsample = 50



models = {
    'SADI': model_sadi_pbm,
}
partial_bm_config['features'] = 'all'
partial_bm_config['length_range'] = (30,30)
mse_folder = f"results_agaid_qual_pbm_{name}/metric"
data_folder = f"results_agaid_qual_pbm_{name}/data"





pbm = [1,3,5,7,9,11]
for bm in pbm:  
    partial_bm_config['features'] = bm
    print(f"features: {partial_bm_config['features']}")
    evaluate_imputation_all(models=models, trials=20, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, test_indices=[32,33], mean=mean, std=std, partial_bm_config=partial_bm_config)
#     evaluate_imputation_all(models=models, trials=1, mse_folder=data_folder, dataset_name='agaid', batch_size=1, test_indices=[32,33], mean=mean, std=std, partial_bm_config=partial_bm_config, data=True, unnormalize=True)

# lengths = [50, 100, 150]
# # LTE50, dept index start: 9
# feat = 'LTE50'
# print(f"\nBlackout:\n")
# for l in lengths:
#     print(f"length = {l}")
#     evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, length=l, test_indices=[32,33], mean=mean, std=std)
    # evaluate_imputation(models, data_folder, length=l, trials=1, data=True)
    # if l == 100:
    #     evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='agaid', length=l, trials=1, batch_size=1, test_indices=[33], data=True, mean=mean, std=std, unnormalize=True) #, exclude_features=feature_dependency[feat])

# lengths = [50, 100, 150]
# for l in lengths:
#     print(f"\nForecasting = {l}")
#     evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, length=l, forecasting=True, test_indices=[32,33], mean=mean, std=std)
# # #     # evaluate_imputation(models, mse_folder=data_folder, length=l, forecasting=True, trials=1, data=True)
# #     evaluate_imputation_all(models=models, mse_folder=data_folder, forecasting=True, dataset_name='agaid', length=100, trials=1, batch_size=1, test_indices=[33], data=True, mean=mean, std=std)

miss_ratios = [0.2, 0.5, 0.8]
for ratio in miss_ratios:
    print(f"\nRandom Missing: ratio ({ratio})\n")
    evaluate_imputation_all(models=models, trials=10, mse_folder=mse_folder, dataset_name='agaid', batch_size=16, missing_ratio=ratio, random_trial=True, test_indices=[32,33], mean=mean, std=std, rmse=True)
    # evaluate_imputation(models, mse_folder=data_folder, random_trial=True, trials=1, data=True, missing_ratio=ratio)
    # evaluate_imputation_all(models=models, mse_folder=data_folder, dataset_name='agaid', trials=1, batch_size=1, data=True, missing_ratio=ratio, test_indices=[33], random_trial=True, mean=mean, std=std)
