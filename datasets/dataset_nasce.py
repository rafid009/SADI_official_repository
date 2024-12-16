import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from config_ablation import partial_bm

def parse_data(sample, rate=0.2, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None, partial_bm_config=None):
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
            if len(indices) != 0:
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
        start_idx = np.random.choice(a)
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
    return obs_data, obs_mask, mask, sample, gt_intact

class NASCE_Dataset(Dataset):
    def __init__(self, filename, mean_std_file, rate=0.1, is_test=False, length=100, seed=10, forward_trial=-1, random_trial=False, pattern=None, partial_bm_config=None, is_valid=False) -> None:
        super().__init__()
        
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        X = np.load(filename)
        self.eval_length = X.shape[1]

        if is_test or is_valid:
            self.mean = np.load(f"{mean_std_file}_mean.npy")
            self.std = np.load(f"{mean_std_file}_std.npy")
        else:
            train_X = X.copy()
            train_X = train_X.reshape((-1, X.shape[2]))
            self.mean = np.nanmean(train_X, axis=0)
            np.save(f"{mean_std_file}_mean.npy", self.mean)

            self.std = np.nanstd(train_X, axis=0)
            np.save(f"{mean_std_file}_std.npy", self.std)
        include_features = []

        for i in range(X.shape[0]):
            obs_val, obs_mask, mask, sample, obs_intact = parse_data(X[i], rate, is_test, length, include_features=include_features, \
                                                                     forward_trial=forward_trial, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config)
            
            self.observed_values.append(obs_val)
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)
            # self.obs_data_intact.append(sample)
            self.gt_intact.append(obs_intact)

        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.observed_masks
        self.gt_intact = ((self.gt_intact - self.mean) / self.std) * self.gt_masks.numpy()

        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
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


def get_dataloader(filename, mean_std_file, batch_size=16, missing_ratio=0.2, is_test=False,is_pattern=False, type='year', data='temps'):
    # np.random.seed(seed=seed)
    train_dataset = NASCE_Dataset(filename, mean_std_file, rate=missing_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if is_pattern:
        test_dataset = NASCE_Dataset(f'./data/nasce/X_OR_{data}_test_{type}.npy', mean_std_file, rate=0.2, pattern=None, is_valid=True)
    else:
        test_dataset = NASCE_Dataset(f'./data/nasce/X_OR_{data}_test_{type}.npy', mean_std_file, rate=missing_ratio, pattern=None, is_valid=True)
    
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    return train_loader, test_loader


def get_testloader_nasce(filename, mean_std_file, n_steps=366, batch_size=16, missing_ratio=0.2, seed=10, length=100, forecasting=False, random_trial=False, pattern=None, partial_bm_config=None):
    np.random.seed(seed=seed)
    if forecasting:
        forward = n_steps - length
        test_dataset = NASCE_Dataset(filename, mean_std_file, rate=missing_ratio, is_test=True, length=length, forward_trial=forward, pattern=pattern, partial_bm_config=partial_bm_config)
    else:
        test_dataset = NASCE_Dataset(filename, mean_std_file, rate=missing_ratio, is_test=True, length=length, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader