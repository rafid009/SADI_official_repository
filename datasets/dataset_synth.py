import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from config_ablation import partial_bm
from datasets.synthetic_data import create_synthetic_data_v2, create_synthetic_data_v3, create_synthetic_data_v4, create_synthetic_data_v5, create_synthetic_data_v6, create_synthetic_data_v7, create_synthetic_data_v8, create_synthetic_data_v1

def parse_data(sample, rate=0.3, is_test=False, length=100, include_features=None, forward_trial=-1, lte_idx=None, random_trial=False, pattern=None, partial_bm_config=None):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    if pattern is not None:
        shp = sample.shape
        choice = np.random.randint(low=pattern['start'], high=(pattern['start'] + pattern['num_patterns'] - 1))
        # print(f"start: {pattern['start']} end: {(pattern['start'] + pattern['num_patterns'] - 1)} choice: {choice}")
        filename = f"{pattern['pattern_dir']}/pattern_{choice}.npy"
        mask = np.load(filename)
        mask = mask * obs_mask
        evals = sample.reshape(-1).copy()
        
        while ((obs_mask == mask).all()):
            choice = np.random.randint(low=pattern['start'], high=(pattern['start'] + pattern['num_patterns'] - 1))
            # print(f"start: {pattern['start']} end: {(pattern['start'] + pattern['num_patterns'] - 1)} choice: {choice}")
            filename = f"{pattern['pattern_dir']}/pattern_{choice}.npy"
            mask = np.load(filename)
        
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
        # obs_data_intact = evals.reshape(shp)
    elif random_trial:
        evals = sample.copy()
        values = evals.copy()
        for i in range(evals.shape[1]):
            indices = np.where(~np.isnan(evals[:, i]))[0].tolist()
            indices = np.random.choice(indices, int(len(indices) * rate))
            values[indices, i] = np.nan
        mask = ~np.isnan(values)
        gt_intact = values
        obs_data = np.nan_to_num(evals, copy=True)
    elif partial_bm_config is not None:
        total_features = np.arange(sample.shape[1])
        features = np.random.choice(total_features, partial_bm_config['features'])
        obs_data, mask, gt_intact = partial_bm(sample, features, partial_bm_config['length_range'], partial_bm_config['n_chunks'])
    elif forward_trial != -1:
        indices = np.where(~np.isnan(sample[:, lte_idx]))[0].tolist()
        start = indices[forward_trial]
        obs_data = np.nan_to_num(sample, copy=True)
        gt_intact = sample.copy()
        gt_intact[start:, :] = np.nan
        mask = ~np.isnan(gt_intact)
    else:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        # print(f"random choice: {start_idx}")
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
    # print(f"\n\nobs data 1: {obs_data}")
    return obs_data, obs_mask, mask, sample, gt_intact

class Synth_Dataset(Dataset):
    def __init__(self, n_steps, n_features, num_seasons, rate=0.1, is_test=False, length=100, exclude_features=None, seed=10, forward_trial=-1, is_mcar=False, is_col_miss=None, random_trial=False, v2='v1', noise=False, mean=None, std=None, pattern=None, partial_bm_config=None) -> None:
        super().__init__()
        self.eval_length = n_steps
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        if v2 == 'v2':
            X, mu, sigma = create_synthetic_data_v2(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v4':
            X, mu, sigma = create_synthetic_data_v4(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v5':
            X, mu, sigma = create_synthetic_data_v5(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v6':
            X, mu, sigma = create_synthetic_data_v6(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v7':
            X, mu, sigma = create_synthetic_data_v7(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v8':
            X, mu, sigma = create_synthetic_data_v8(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v1':
            X, mu, sigma = create_synthetic_data_v1(n_steps, num_seasons, seed=seed, noise=noise)
        elif v2 == 'v3':
            X, mu, sigma = create_synthetic_data_v3(n_steps, num_seasons, seed=seed, noise=noise, is_mcar=is_mcar, is_col_miss=is_col_miss)
        
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = mu
            self.std = sigma
            synth_dir = f"./data/synth/{v2}"
            if not os.path.isdir(synth_dir):
                os.makedirs(synth_dir)
            np.save(f"{synth_dir}/mean.npy", self.mean)
            np.save(f"{synth_dir}/std.npy", self.std)

        include_features = []

        for i in range(X.shape[0]):
            obs_val, obs_mask, mask, sample, obs_intact = parse_data(X[i], rate, is_test, length, include_features=include_features, forward_trial=forward_trial, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config)
            self.observed_values.append(obs_val)
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)
            self.obs_data_intact.append(sample)
            self.gt_intact.append(obs_intact)
        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.obs_data_intact = np.array(self.obs_data_intact)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        # print(f"obs mask: {self.observed_values}\n\nmean: {}")
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.observed_masks
        self.obs_data_intact = ((self.obs_data_intact - self.mean) / self.std) * self.observed_masks.numpy()
        self.gt_intact = ((self.gt_intact - self.mean) / self.std) * self.gt_masks.numpy()

        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            # "gt_mask": self.gt_masks[index],
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


def get_dataloader(n_steps, n_features, num_seasons, batch_size=16, missing_ratio=0.1, seed=10, is_test=False, v2='v1',  noise=False, is_mcar=False, is_col_miss=None):
    np.random.seed(seed=seed)
    train_dataset = Synth_Dataset(n_steps, n_features, num_seasons, rate=missing_ratio, seed=seed, v2=v2, noise=noise, is_mcar=is_mcar, is_col_miss=is_col_miss, is_test=is_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    synth_dir = f"./data/synth/{v2}"
    if os.path.exists(f"{synth_dir}/mean.npy"):
        mean = np.load(f"{synth_dir}/mean.npy")
    else:
        mean = None
    
    if os.path.exists(f"{synth_dir}/std.npy"):
        std = np.load(f"{synth_dir}/std.npy")
    else:
        std = None
    test_dataset = Synth_Dataset(n_steps, n_features, 10, rate=missing_ratio, seed=seed*5, v2=v2, noise=noise, mean=mean, std=std, is_mcar=is_mcar, is_col_miss=is_col_miss)
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader, mean, std

def get_testloader(n_steps, n_features, num_seasons, missing_ratio=0.2, seed=10, exclude_features=None, length=100, forward_trial=False, random_trial=False, v2='v1', noise=False):
    np.random.seed(seed=seed)
    if forward_trial:
        forward = n_steps - length
        test_dataset = Synth_Dataset(n_steps, n_features, num_seasons, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, seed=seed, forward_trial=forward, v2=v2, noise=noise)
    else:
        test_dataset = Synth_Dataset(n_steps, n_features, num_seasons, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, seed=seed, random_trial=random_trial, v2=v2, noise=noise)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return test_loader

def get_testloader_synth(n_steps, n_features, num_seasons, batch_size=16, missing_ratio=0.2, seed=10, exclude_features=None, length=100, forecasting=False, random_trial=False, v2='v1', noise=False, mean=None, std=None, pattern=None, partial_bm_config=None):
    np.random.seed(seed=seed)
    if forecasting:
        forward = n_steps - length
        test_dataset = Synth_Dataset(n_steps, n_features, num_seasons, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, seed=seed, forward_trial=forward, v2=v2, noise=noise, mean=mean, std=std, pattern=pattern, partial_bm_config=partial_bm_config)
    else:
        test_dataset = Synth_Dataset(n_steps, n_features, num_seasons, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features, seed=seed, random_trial=random_trial, v2=v2, noise=noise, mean=mean, std=std, pattern=pattern, partial_bm_config=partial_bm_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader