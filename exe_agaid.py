from models.sadi_wrapper import SADI_Agaid
from datasets.dataset_agaid import get_dataloader
from utils.utils import *
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
from models.brits_util import *
# from models.brits import BRITSModel as BRITS

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(10, 100)

miss_type = 'random'

# config_dict_csdi_pattern = {
#     'train': {
#         'epochs': 3000,
#         'batch_size': 8,
#         'lr': 1.0e-4
#     },      
#     'diffusion': {
#         'layers': 4, 
#         'channels': 64,
#         'nheads': 8,
#         'diffusion_embedding_dim': 128,
#         'beta_start': 0.0001,
#         'beta_end': 0.5,
#         'num_steps': 70,
#         'schedule': "quad",
#         'is_fast': False,
#     },
#     'model': {
#         'is_unconditional': 0,
#         'timeemb': 128,
#         'featureemb': 16,
#         'target_strategy': miss_type,
#         'type': 'CSDI',
#         'n_layers': 3, 
#         'd_time': 100,
#         'n_feature': len(features),
#         'd_model': 128,
#         'd_inner': 128,
#         'n_head': 8,
#         'd_k': 64,
#         'd_v': 64,
#         'dropout': 0.1,
#         'diagonal_attention_mask': True,
#         'num_patterns': 15000,
#         'num_val_patterns': 5000,
#         'pattern_dir': './data/AgAid/miss_pattern'
#     },
# }

# config_dict_csdi_random = {
#     'train': {
#         'epochs': 5000,
#         'batch_size': 8,
#         'lr': 1.0e-4
#     },      
#     'diffusion': {
#         'layers': 4, 
#         'channels': 64,
#         'nheads': 8,
#         'diffusion_embedding_dim': 128,
#         'beta_start': 0.0001,
#         'beta_end': 0.5,
#         'num_steps': 50,
#         'schedule': "quad",
#         'is_fast': False,
#     },
#     'model': {
#         'is_unconditional': 0,
#         'timeemb': 128,
#         'featureemb': 16,
#         'target_strategy': "random",
#         'type': 'CSDI',
#         'n_layers': 3, 
#         'd_time': 252,
#         'n_feature': len(features),
#         'd_model': 128,
#         'd_inner': 128,
#         'n_head': 4,
#         'd_k': 64,
#         'd_v': 64,
#         'dropout': 0.1,
#         'diagonal_attention_mask': False,
#         'num_patterns': 15000,
#         'num_val_patterns': 5000,
#         'pattern_dir': './data/AgAid/miss_pattern'
#     }
# }

# # dataset_name = 'agaid' + '_weather'
data_file = './data/AgAid/ColdHardiness_Grape_Merlot_2.csv'
# config_dict_csdi = config_dict_csdi_pattern if miss_type.startswith('pattern') else config_dict_csdi_random

train_loader, valid_loader, mean, std = get_dataloader(
    seed=seed,
    filename=data_file,
    batch_size=16,
    missing_ratio=0.2,
    season_idx=[32, 33]
)

# # np.save('agaid_mean.npy', mean)
# # np.save('agaid_std.npy', std)
# model_csdi = CSDI_Agaid(config_dict_csdi, device).to(device)
model_folder = "./saved_model_agaid"
# if not os.path.isdir(model_folder):
#     os.makedirs(model_folder)
# filename = f'model_csdi_{miss_type}.pth'
# train(
#     model_csdi,
#     config_dict_csdi["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=filename
# )
# nsample = 50
# model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# print(f"CSDI params: {get_num_params(model_csdi)}")
# evaluate(model_csdi, valid_loader, nsample=nsample, scaler=1, foldername=model_folder)

    
config_dict_diffsaits = {
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
config_dict_diffsaits['ablation'] = common_config['ablation']
config_dict_diffsaits['model']['n_layers'] = 3 # 3 #common_config['n_layers']
config_dict_diffsaits['ablation']['is_2nd_block'] = True
config_dict_diffsaits['ablation']['is_fde'] = True
config_dict_diffsaits['ablation']['fde-time-pos-enc'] = False
config_dict_diffsaits['ablation']['weight_combine'] = True #True
config_dict_diffsaits['ablation']['fde-layers'] = 3
config_dict_diffsaits['name'] = 'normal' # 'skip_fde_1st_mask_pos_enc_loss_p_bm' # just remove _new_loss
# config_dict_diffsaits['name'] = common_config['name']
name = config_dict_diffsaits['name'] #+ "_weather"
print(config_dict_diffsaits)
# model_diff_saits_simple = CSDI_Agaid(config_dict, device, is_simple=True).to(device)
model_sadi = SADI_Agaid(config_dict_diffsaits, device, is_simple=False).to(device)
filename = f'model_diff_saits_{name}.pth'

# model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# 
train(
    model_sadi,
    config_dict_diffsaits["train"],
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
print(f"DiffSAITS params: {get_num_params(model_sadi)}")


config_dict_sadi_pbm = {
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
# model_sadi_pbm.load_state_dict(torch.load(f"{model_folder}/{filename}"))


###################  SAITS  ###################

X = None
for j, train_batch in enumerate(train_loader, start=1):
    if X is None:
        X = train_batch['observed_data']
    else:
        X = np.concatenate([X, train_batch['observed_data']], axis=0)


# # # observed_mask = ~np.isnan(X)

saits_model_file = f"{model_folder}/model_saits.pth"
saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=800, device=device)

saits.fit({"X":X}) 
pickle.dump(saits, open(saits_model_file, 'wb'))
# saits = pickle.load(open(saits_model_file, 'rb'))


###################  SAITS  ###################

###################  MICE ###################
mice_model_file = f"{model_folder}/model_mice.pth"
mice = IterativeImputer(random_state=100, max_iter=30)
mice.fit(np.reshape(X, (-1, len(features))))
pickle.dump(mice, open(mice_model_file, 'wb'))

# mice = pickle.load(open(mice_model_file, 'rb'))

###################  BRITS  ###################
n_epochs = 3000
RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 1
# LABEL_WEIGHT = 1

brits_model_path_name = 'BRITS'
brits_model_path = f'{model_folder}/model_{brits_model_path_name}.model'#synth_{n_random}.model'
brits_model = BRITS(n_steps=252, n_features=len(features), rnn_hidden_size=RNN_HID_SIZE, batch_size=32, epochs=n_epochs, patience=800, device=device)
# brits_model = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, feature_len=len(features), seq_len=252)
brits_model.fit({"X": X})

# pickle.dump(brits_model, open(brits_model_path, 'wb'))
# brits_model = pickle.load(open(brits_model_path, 'rb'))

# fs = './json/json_agaid'

# prepare_brits_input(fs, X)
# if torch.cuda.is_available():
#     brits_model = brits_model.cuda()
# brits_model = brits_train(brits_model, n_epochs, 32, brits_model_path, data_file=fs)
# brits_model.load_state_dict(torch.load(brits_model_path))



models = {
    # 'CSDI': model_csdi,
    'SAITS': saits,
    # 'KNN': knn,
    'SADI': model_sadi_pbm,
    # 'SADI_pbm': 
    'BRITS': brits_model,
    'MICE': mice
}
partial_bm_config['features'] = 'all'
partial_bm_config['length_range'] = (30,30)
mse_folder = f"results_agaid_qual_pbm_{name}/metric"
data_folder = f"results_agaid_qual_pbm_{name}/data"


# test_patterns_start = 15001
# num_test_patterns = 5000

# test_pattern_config = {
#     'start': test_patterns_start,
#     'num_patterns': num_test_patterns,
#     'pattern_dir': './data/AgAid/miss_pattern'
# }


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


# print("For All")
# for l in lengths:
#     print(f"For length: {l}")
#     # evaluate_imputation(models, mse_folder, length=l, trials=1, season_idx=33)
#     print(f"Blackout Missing:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=10, season_idx=33)
#     evaluate_imputation(models, data_folder, length=l, trials=1, data=True)
#     print(f"Forecasting:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=1, season_idx=33, forecasting=True)
#     evaluate_imputation(models, mse_folder=data_folder, length=l, forecasting=True, trials=1, data=True)
#     print(f"Random Missing:\n")
#     evaluate_imputation(models, mse_folder, length=l, trials=10, season_idx=33, random_trial=True)
#     evaluate_imputation(models, mse_folder=data_folder, length=l, random_trial=True, trials=1, data=True)

    # evaluate_imputation_data(models, length=l)

# feature_combinations = {
    # "temp": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT"],
    # "hum": ["AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY", "MAX_REL_HUMIDITY"],
    # "dew": ["AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT"],
    # "pinch": ["P_INCHES"],
    # "wind": ["WS_MPH", "MAX_WS_MPH"],
    # "sr": ["SR_WM2"],q
    # "leaf": ["LW_UNITY"],
    # "et": ["ETO", "ETR"],
    # "st": ["ST8", "MIN_ST8", "MAX_ST8"],
#     "temp-hum": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY"],
#     "temp-hum-dew": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT"],
#     "for-lte": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-temp": ["AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY", "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT",
#          "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-hum": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", 
#          "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-dew": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-et": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "SR_WM2", "LW_UNITY", "ST8", "MIN_ST8", "MAX_ST8"],
#     "for-sr": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT", "AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY",
#          "MAX_REL_HUMIDITY", "AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT", "P_INCHES", "WS_MPH", "MAX_WS_MPH",
#          "LW_UNITY", "ETO", "ETR", "ST8", "MIN_ST8", "MAX_ST8"]
# }
# print(f"The exclusions")
# for key in feature_combinations.keys():
#     for l in lengths:
#         print(f"For length: {l}")
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=1)
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=20)
        # evaluate_imputation_data(models, exclude_key=key, exclude_features=feature_combinations[key], length=l)
# forward_evaluation(models, filename, features)

# input_file = "ColdHardiness_Grape_Merlot_2.csv"

# cross_validate(input_file, config_dict_csdi, config_dict_diffsaits, seed=10)