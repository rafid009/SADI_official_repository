import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import pandas as pd
import math
torch.set_printoptions(precision=10)

def dynamic_sensor_selection(X_train, X_loc, rate=0.5, L=30, K=2):
    inverse_rate = int((1 - rate) * X_train.reshape(L, -1, K).shape[1])
    indices = np.random.choice(X_train.reshape(L, -1, K).shape[1], inverse_rate, replace=False)
    shp = X_train.shape
    X_train = X_train.reshape(X_train.shape[0], -1, K)
    X_train[:, indices, :] = np.nan
    X_loc[indices, :] = 0
    X_train = X_train.reshape(shp)
    return X_train, X_loc


def parse_data(sample, rate=0.2):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    # if not is_test:
    shp = sample.shape
    evals = sample.reshape(-1).copy()
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * rate))
    values = evals.copy()
    values[indices] = np.nan
    mask = ~np.isnan(values)
    mask = mask.reshape(shp)
    # gt_intact = values.reshape(shp).copy()
    obs_data = np.nan_to_num(evals, copy=True)
    obs_data = obs_data.reshape(shp)

    return obs_data, obs_mask, mask

def get_train_data(train_indices, X, X_loc):
    X_real = np.zeros(X.shape) # B, L, N, K
    X_real[:, :, :len(train_indices), :] = X[:, :, train_indices, :]

    X_loc_real = np.zeros(X_loc.shape) # B, N, 3
    X_loc_real[:len(train_indices), :] = X_loc[train_indices, :]
    return X_real, X_loc_real


def get_test_data(X_train, X, X_loc_train, X_loc, index, train_indices):
    if index in train_indices:
        X_test = X_train.copy()
        X_test = X_test.reshape(X_test.shape[0], -1, 2)
        X_test_values = X_test.copy()
        X_test_values[:, index, :] = np.nan
        X_test_loc = X_loc_train.copy()
    else:
        X_test = X_train.copy() # L, N*K
        X_test = X_test.reshape(X_test.shape[0], -1, 2) # L, N, K
        X = X.reshape(X.shape[0], -1, 2)
        X_test[:, len(train_indices), :] = X[:, index, :]
        X_test_values = X_test.copy()
        X_test_values[:, len(train_indices), :] = np.nan
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test_values = X_test_values.reshape(X_test_values.shape[0], -1)

        X_test_loc = X_loc_train.copy()
        X_test_loc = X_test_loc.reshape(-1, 3)
        X_loc = X_loc.reshape(-1, 3)
        X_test_loc[len(train_indices), :] = X_loc[index, :]
        X_test_loc = X_test_loc.reshape(-1)
    return X_test, X_test_values, X_test_loc

def get_test_data_spatial(X_train, X_test, X_loc_train, X_loc_test, index, X_pristi, deltas=False, test_train_indices=None):
    # print(f"X_train: {X_train.shape}")
    X_train = X_train.reshape(X_train.shape[0], -1, 2)
    if test_train_indices is not None:
        X_test = X_test.reshape(X_test.shape[0], -1, 2)
        X_test_temp = X_test[:, test_train_indices, :]

        X_train = X_test_temp
        X_loc_train = X_loc_test[test_train_indices, :]
        
    if isinstance(index, int): 
        X_test_missing = np.expand_dims(X_test.reshape(X_test.shape[0], -1, 2)[:, index,:], axis=1)
    else:
        X_test_missing = X_test.reshape(X_test.shape[0], -1, 2)[:, index,:]
    X_pristi = X_pristi.reshape(X_pristi.shape[0], -1, 2)
    X_pristi[:, X_train.shape[1] - 1 + index, :] = X_test.reshape(X_test.shape[0], -1, 2)[:,index,:]
    
    if isinstance(index, int): 
        X_loc_test_missing = np.expand_dims(X_loc_test[index,:], axis=0)
    else:
        X_loc_test_missing = X_loc_test[index,:]
    
    values = X_train.copy()
    if deltas:
        X_loc_train = X_loc_train - X_loc_test_missing

    values_pristi = X_pristi.copy()
    values_pristi[:, X_train.shape[1] - 1 + index, :] = np.nan

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test_missing = X_test_missing.reshape(X_test_missing.shape[0], -1)
    values = values.reshape(X_train.shape[0], -1)
    return X_train, values, X_loc_train, X_pristi, values_pristi, X_test_missing, X_loc_test_missing



def get_location_index(X_loc, loc):
    index = 0
    for loc_x in X_loc:
        if loc_x[0] == loc[0] and loc_x[1] == loc[1] and loc_x[2] == loc[2]:
            break
        index += 1
    return index 


def parse_data_spatial(sample, X_loc, X_test_loc, neighbor_location, spatial_choice=None, is_separate=False, index=None):
    
    
    L, K = sample.shape
    evals = sample.copy().reshape(L, -1, 2)

    if index is None:
        chosen_location = np.random.choice(np.arange(X_test_loc.shape[0]))
    else:
        chosen_location = index
    location_idx = get_location_index(X_loc, X_test_loc[chosen_location])

    neighbors = json.load(open(neighbor_location, 'r'))

    
    locations = neighbors[f"{location_idx}"]
    
    
    evals_pristi = np.zeros(evals.shape)
    evals_pristi[:, locations, :] = evals[:, locations, :]
    obs_mask_pristi = ~np.isnan(evals_pristi)

    values = evals.copy()
    if is_separate:
        missing_data = values[:, location_idx, :]
        missing_data_mask = ~np.isnan(missing_data)
        missing_data = np.nan_to_num(missing_data, copy=True)
        values[:, location_idx, :] = np.nan
    else: 
        values[:, location_idx, :] = np.nan
        values = values[:, locations, :]
        mask = ~np.isnan(values)
    
    
    evals = evals[:, locations, :]
    obs_mask = ~np.isnan(evals)
    if is_separate:
        mask = obs_mask

    values_pristi = evals_pristi.copy()
    values_pristi[:, location_idx, :] = np.nan
    
    
    mask_pristi = ~np.isnan(values_pristi)

    evals = evals.reshape(L, -1)
    evals = np.nan_to_num(evals)
    obs_mask = obs_mask.reshape(L, -1)
    mask = mask.reshape(L, -1)

    evals_pristi = evals_pristi.reshape(L, -1)
    evals_pristi = np.nan_to_num(evals_pristi)
    obs_mask_pristi = obs_mask_pristi.reshape(L, -1)
    mask_pristi = mask_pristi.reshape(L, -1)

    evals_loc = X_loc[locations]
  
    missing_locs = np.expand_dims(X_test_loc[chosen_location], axis=0)
    if is_separate:
        return evals, obs_mask, mask, evals_loc, evals_pristi, mask_pristi, obs_mask_pristi, missing_locs, values, missing_data, missing_data_mask, locations
    else:
        return evals, obs_mask, mask, evals_loc, evals_pristi, mask_pristi, obs_mask_pristi, missing_locs, values, locations

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance between two lat/lon points in meters.
    """
    R = 6371000  # Earth radius (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

class SWaT_Dataset(Dataset):
    def __init__(self, data, mean_std_file, n_features, rate=0.2, is_test=False, is_valid=False) -> None:
        super().__init__()
        
        self.observed_values = []
        self.observed_values_pristi = []
        self.spatial_info = []
        # self.obs_data_intact = []
        self.observed_masks = []
        self.observed_masks_pristi = []
        self.gt_masks = []
        self.gt_masks_pristi = []
        self.total_loc = []
        self.gt_intact = []
        self.is_test = is_test or is_valid
        self.is_valid = is_valid

        X = data
        B, L, K = X.shape
        
        
        X = X.reshape(B, L, -1)

        B, L, K = X.shape
        

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

        
        for i in tqdm(range(X.shape[0])):
            
                
            obs_val, obs_mask, mask = parse_data(X[i], rate)
            self.observed_values.append(obs_val)
            
            
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)
       
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        print(f"obs_values nan: {torch.isnan(self.observed_values).sum()}")
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        self.observed_values = ((self.observed_values.reshape(self.observed_values.shape[0], L, -1) - self.mean) / self.std) * self.observed_masks.reshape(self.observed_masks.shape[0], L, -1)
        if is_test or is_valid:
            self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        
        # self.neighbor_location = None #"./data/nacse/neighbors.json"

           
        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index].reshape(self.observed_values[index].shape[0], -1),
            "observed_mask": self.observed_masks[index].reshape(self.observed_masks[index].shape[0], -1),
            "timepoints": np.arange(self.eval_length)
        }
        if len(self.gt_masks) != 0:
            s["gt_mask"] = self.gt_masks[index].reshape(self.gt_masks[index].shape[0], -1)
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(mean_std_file, n_features, batch_size=16, missing_ratio=0.2, is_test=False):
    np.random.seed(seed=100)
    input_folder = './data/swat'
    normal_data = np.load(f"{input_folder}/SWaT_minute_segments_normal.npy")
    train_data = normal_data[:int(0.8 * normal_data.shape[0])]
    test_data = normal_data[int(0.8 * normal_data.shape[0]):]
    train_dataset = SWaT_Dataset(train_data, mean_std_file, n_features, rate=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SWaT_Dataset(test_data, mean_std_file, n_features, rate=missing_ratio, is_valid=True)
    
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    return train_loader, test_loader


def get_testloader_swat(total_stations, mean_std_file, n_features, n_steps=366, batch_size=16, missing_ratio=0.2, seed=10):
    np.random.seed(seed=seed)
    input_folder = './data/swat'
    data = np.load(f"{input_folder}/SWaT_minute_segments.npy")
    test_dataset = SWaT_Dataset(total_stations, mean_std_file, n_features, rate=missing_ratio)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader