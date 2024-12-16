from datasets.process_data import *
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from config_ablation import partial_bm

def parse_data(sample, rate=0.2, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None,  partial_bm_config=None):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    if pattern is not None:
        shp = sample.shape
        choice = np.random.randint(low=pattern['start'], high=(pattern['start'] + pattern['num_patterns'] - 1))
        filename = f"{pattern['pattern_dir']}/pattern_{choice}.npy"
        mask = np.load(filename)
        mask = mask * obs_mask
        evals = sample.reshape(-1).copy()
    
        
        eval_mask = mask.reshape(-1).copy()
        gt_indices = np.where(eval_mask)[0].tolist()
        miss_indices = np.random.choice(
            gt_indices, (int)(len(gt_indices) * rate), replace=False
        )
        gt_intact = sample.reshape(-1).copy()
        gt_intact[miss_indices] = np.nan
        gt_intact = gt_intact.reshape(shp)
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
    elif not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        gt_intact = values.reshape(shp).copy()
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
    elif random_trial:
        evals = sample.copy()
        values = evals.copy()
        for i in range(evals.shape[1]):
            indices = np.where(~np.isnan(evals[:, i]))[0].tolist()
            indices = np.random.choice(indices, int(len(indices) * rate), replace=False)
            values[indices, i] = np.nan
        mask = ~np.isnan(values)
        gt_intact = values
        obs_data = np.nan_to_num(evals, copy=True)
    elif forward_trial != -1:
        indices = np.where(~np.isnan(sample[:, lte_idx]))[0].tolist()
        start = indices[forward_trial]
        obs_data = np.nan_to_num(sample, copy=True)
        gt_intact = sample.copy()
        gt_intact[start:, :] = np.nan
        mask = ~np.isnan(gt_intact)
    elif partial_bm_config is not None:
        total_features = np.arange(sample.shape[1])
        features = np.random.choice(total_features, partial_bm_config['features'], replace=False)
        obs_data, mask, gt_intact = partial_bm(sample, features, partial_bm_config['length_range'], partial_bm_config['n_chunks'])
    else:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        print(f"start index choice: {start_idx}")
        end_idx = start_idx + length
        obs_data_intact = sample.copy()
        if include_features is None or len(include_features) == 0:
            obs_data_intact[start_idx:end_idx, :] = np.nan
        else:
            obs_data_intact[start_idx:end_idx, include_features] = np.nan
        mask = ~np.isnan(obs_data_intact)
        gt_intact = obs_data_intact
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        # obs_intact = np.nan_to_num(obs_intact, copy=True)
    return obs_data, obs_mask, mask, sample, gt_intact




def get_mask_mnr(sample, rate):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    flatten_sample = np.reshape(sample, -1)
    k = int(len(flatten_sample) * rate)
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = np.random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, rate):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""
    flatten_sample = np.reshape(sample, -1)
    k = int(len(flatten_sample) * rate)
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = np.random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


class Agaid_Dataset(Dataset):
    def __init__(self, X, mean, std, eval_length=252, rate=0.2, is_test=False, length=100, exclude_features=None, forward_trial=-1, lte_idx=None, randon_trial=False, pattern=None, partial_bm_config=None) -> None:
        super().__init__()
        self.eval_length = eval_length
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        
        self.mean = torch.from_numpy(mean) 
        self.std = torch.from_numpy(std) 

        
        include_features = []
        if exclude_features is not None:
            for i in range(len(features)):
                if features[i] not in exclude_features:
                    include_features.append(i)

        for i in range(len(X)):
            obs_data, obs_mask, gt_mask, obs_data_intact, gt_intact_data = parse_data(X[i], rate=rate, is_test=is_test, length=length, include_features=include_features, forward_trial=forward_trial, lte_idx=lte_idx, random_trial=randon_trial, pattern=pattern, partial_bm_config=partial_bm_config)
            self.obs_data_intact.append(obs_data_intact)
            self.gt_masks.append(gt_mask)
            self.observed_values.append(obs_data)
            self.observed_masks.append(obs_mask)
            self.gt_intact.append(gt_intact_data)
        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.obs_data_intact = np.array(self.obs_data_intact)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.observed_masks
        self.obs_data_intact = ((self.obs_data_intact - self.mean.numpy()) / self.std.numpy()) * self.observed_masks.numpy()
        self.gt_intact = ((self.gt_intact - self.mean.numpy()) / self.std.numpy()) * self.gt_masks.numpy()


        # print(f"obs_val: {self.observed_values.shape}")

    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "obs_data_intact": self.obs_data_intact[index],
            "timepoints": np.arange(self.eval_length),
            "gt_intact": self.gt_intact[index]
        }
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
        else:
            s["gt_mask"] = self.gt_masks[index]
        return s
    
    def __len__(self):
        return len(self.observed_values)

        
def get_dataloader(filename='./data/agaid/ColdHardiness_Grape_Merlot_2.csv', batch_size=16, missing_ratio=0.2, seed=10, is_test=False, season_idx=-1, random_trial=False):
    np.random.seed(seed=seed)
    df = pd.read_csv(filename)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)

    X, Y = split_XY(season_df, max_length, season_array, features)

    train_X = []
    test_X = X[season_idx]
    if len(test_X.shape) != 3:
        test_X = np.expand_dims(test_X, 0)
    for i in range(len(X)):
        if i != season_idx:
            train_X.append(X[i])
    train_X = np.array(train_X)
    train_X_real = np.reshape(train_X, (-1, len(features)))
    mean, std = np.nanmean(train_X_real, axis=0), np.nanstd(train_X_real, axis=0)

    train_dataset = Agaid_Dataset(train_X, mean, std, rate=missing_ratio)
    test_dataset = Agaid_Dataset(test_X, mean, std, rate=missing_ratio, is_test=is_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader, mean, std

def get_testloader(filename='./data/agaid/ColdHardiness_Grape_Merlot_2.csv', missing_ratio=0.2, seed=10, season_idx=None, exclude_features=None, length=100, forward_trial=-1, lte_idx=None, random_trial=False, forecastig=False):
    np.random.seed(seed=seed)
    df = pd.read_csv(filename)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
    if season_idx is not None:
        train_season_df = season_df.drop(season_array[season_idx], axis=0)
    else:
        train_season_df = season_df.drop(season_array[-1], axis=0)
        train_season_df = train_season_df.drop(season_array[-2], axis=0)
    mean, std = get_mean_std(train_season_df, features)

    X, Y = split_XY(season_df, max_length, season_array, features)
    X = np.expand_dims(X[season_idx], 0)
    if forecastig:
        forward_trial = max_length - length
    test_dataset = Agaid_Dataset(X, mean, std, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, forward_trial=forward_trial, lte_idx=lte_idx, randon_trial=random_trial)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return test_loader

def get_forward_testloader(X, mean, std, forward_trial=-1, lte_idx=None):  
    test_dataset = Agaid_Dataset(X, mean, std, is_test=True, forward_trial=forward_trial, lte_idx=lte_idx)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return test_loader

def get_testloader_agaid(filename='./data/agaid/ColdHardiness_Grape_Merlot_2.csv', batch_size=16, missing_ratio=0.2, seed=10, exclude_features=None, length=100, forward_trial=-1, random_trial=False, forecastig=False, mean=None, std=None, test_idx=-1, pattern=None, partial_bm_config=None):
    np.random.seed(seed=seed)
    df = pd.read_csv(filename)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)

    X, Y = split_XY(season_df, max_length, season_array, features)
    X = X[test_idx]
    if len(X.shape) < 3:
        X = np.expand_dims(X, 0)
    if forecastig:
        forward_trial = max_length - length
    test_dataset = Agaid_Dataset(X, mean, std, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, forward_trial=forward_trial, randon_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader