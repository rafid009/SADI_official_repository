import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from json import JSONEncoder
import os
from datasets.dataset_agaid import get_testloader, get_testloader_agaid
from datasets.dataset_synth import get_testloader_synth


from models.main_model import SADI_Agaid
from pypots.imputation import SAITS
import math
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=torch.inf)

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    filename=""
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        output_path = foldername + f"/{filename if len(filename) != 0 else 'model_sadi.pth'}"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    model.train()
    for epoch_no in range(config["epochs"]):
        avg_loss = 0

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            torch.save(model.state_dict(), output_path)
            model.train()

    if filename != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def evaluate_imputation_all(models, mse_folder, dataset_name='', batch_size=16, trials=3, length=-1, random_trial=False, forecasting=False, missing_ratio=0.01, test_indices=None, data=False, noise=False, filename=None, is_yearly=True, n_steps=366, pattern=None, mean=None, std=None, partial_bm_config=None, exclude_features=None, unnormalize=False):  
    nsample = 50
    if 'CSDI' in models.keys():
        models['CSDI'].eval()
    if 'SADI' in models.keys():
        models['SADI'].eval()

    results_trials_mse = {'csdi': {}, 'sadi': {}, 'saits': {}, 'knn': {}, 'mice': {}, 'brits': {}}
    results_trials_mae = {'csdi': {}, 'sadi': {}, 'saits': {}, 'mice': {}, 'brits': {}}
    results_mse = {'csdi': 0, 'sadi': 0, 'sadi_median': 0, 'sadi_mean_med': 0, 'saits': 0, 'knn': 0, 'mice': 0, 'brits': 0}
    results_mae = {'csdi': 0, 'sadi': 0, 'sadi_median': 0, 'saits': 0}
    results_crps = {
        'csdi_trials':{}, 'csdi': 0, 
        'sadi_trials': {}, 'sadi': 0, 'sadi_mean_med': 0,
        }
    results_data = {}
    
    if forecasting and not data and isinstance(length, tuple):
            range_len = (length[0], length[1])
    else:
        range_len = None
    if data:
        trials = 1
    s = 10 #np.random.randint(0,100)
    for trial in range(trials):
        if forecasting and not data and range_len is not None:
            length = np.random.randint(low=range_len[0], high=range_len[1] + 1)
        
        if dataset_name == 'synth_v1':
            test_loader = get_testloader_synth(n_steps=100, n_features=3, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v1', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v2':
            test_loader = get_testloader_synth(n_steps=100, n_features=3, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v2', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v3':
            test_loader = get_testloader_synth(n_steps=100, n_features=4, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v3', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v4':
            test_loader = get_testloader_synth(n_steps=100, n_features=6, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v4', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v5':
            test_loader = get_testloader_synth(n_steps=100, n_features=5, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v5', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v6':
            test_loader = get_testloader_synth(n_steps=100, n_features=4, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v6', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v7':
            test_loader = get_testloader_synth(n_steps=100, n_features=6, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v7', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        elif dataset_name == 'synth_v8':
            test_loader = get_testloader_synth(n_steps=100, n_features=3, batch_size=batch_size, num_seasons=16, seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting, v2='v9', noise=noise, mean=mean, std=std, partial_bm_config=partial_bm_config)
        else:
            test_loader = get_testloader_agaid(seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecastig=forecasting, batch_size=batch_size, test_idx=test_indices, mean=mean, std=std, partial_bm_config=partial_bm_config)
        
        csdi_rmse_avg = 0
        sadi_rmse_avg = 0
        sadi_median_avg = 0
        sadi_mean_med_avg = 0
        saits_rmse_avg = 0
        knn_rmse_avg = 0
        mice_rmse_avg = 0
        brits_rmse_avg = 0

        csdi_mae_avg = 0
        sadi_mae_avg = 0
        saits_mae_avg = 0

        csdi_crps_avg = 0
        sadi_crps_avg = 0

        
        for j, test_batch in enumerate(test_loader, start=1):
            if 'CSDI' in models.keys():
                output = models['CSDI'].evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
            
            if 'SADI' in models.keys():
                output_sadi = models['SADI'].evaluate(test_batch, nsample)
                if 'CSDI' not in models.keys():
                    samples_sadi, c_target, eval_points, observed_points, obs_intact, gt_intact = output_sadi
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                else:
                    samples_sadi, _, _, _, _, _, _ = output_sadi
                samples_sadi = samples_sadi.permute(0, 1, 3, 2)
                samples_sadi_median = samples_sadi.median(dim=1)
                samples_sadi_mean = samples_sadi.mean(dim=1)
                samples_sadi_mean_med = torch.mean(torch.stack([samples_sadi_median.values, samples_sadi_mean], dim=1), dim=1)

            if 'SAITS' in models.keys():
                gt_intact = gt_intact.squeeze(axis=0)
                saits_X = gt_intact #test_batch['obs_data_intact']
                if batch_size == 1:
                    saits_X = saits_X.unsqueeze(0)
                saits_output = models['SAITS'].impute({'X': saits_X})

            if 'KNN' in models.keys():
                gt_intact = gt_intact.squeeze(axis=0)
                knn_X = gt_intact #test_batch['obs_data_intact']
                if batch_size == 1:
                    knn_X = knn_X.unsqueeze(0)
                knn_output = None
                for k in range(knn_X.shape[0]):
                    knn_pred = models['KNN'].transform(knn_X[k])
                    if knn_output is None:
                        knn_output = knn_pred
                    else:
                        knn_output = np.stack([knn_output, knn_pred], axis=0)

            if 'BRITS' in models.keys():
                # brits_output = test_evaluate(models['BRITS'], f'json/json_eval_{dataset_name}', test_batch['observed_data'], c_target, observed_points, eval_points)
                gt_intact = gt_intact.squeeze(axis=0)
                brits_X = gt_intact #test_batch['obs_data_intact']
                if batch_size == 1:
                    brits_X = brits_X.unsqueeze(0)
                brits_output = models['BRITS'].impute({'X': brits_X})

            if 'MICE' in models.keys():
                gt_intact = gt_intact.squeeze(axis=0)
                mice_X = gt_intact
                if batch_size == 1:
                    mice_X = mice_X.unsqueeze(0)
                mice_output = None
                for k in range(mice_X.shape[0]):
                    mice_pred = models['MICE'].transform(mice_X[k].cpu())
                    if mice_output is None:
                        mice_output = np.expand_dims(mice_pred, axis=0)
                    else:
                        # print(f"mice out: {mice_output.shape}\nmice pred: {mice_pred.shape}")
                        mice_pred = np.expand_dims(mice_pred, axis=0)
                        mice_output = np.concatenate([mice_output, mice_pred], axis=0)
            
            if unnormalize:
                if dataset_name == 'pm25':
                    path = "./data/pm25/pm25_meanstd.pk"
                    with open(path, "rb") as f:
                        train_mean, train_std = pickle.load(f)
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                elif dataset_name == 'electricity':
                    path_mean = "./data/Electricity/mean.npy"
                    path_std = "./data/Electricity/std.npy"
                    train_mean = np.load(path_mean)
                    train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                    train_std = np.load(path_std)
                    train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                elif dataset_name == 'nasce':
                    path_mean = f"{filename[1]}_mean.npy"
                    path_std = f"{filename[1]}_std.npy"
                    train_mean = np.load(path_mean)
                    train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                    train_std = np.load(path_std)
                    train_std = torch.tensor(train_std, dtype=torch.float32, device=device)
                else:
                    train_mean = torch.tensor(mean, dtype=torch.float32, device=device)
                    train_std = torch.tensor(std, dtype=torch.float32, device=device)

                if 'CSDI' in models.keys():
                    samples_median_csdi = (samples_median.values * train_std) + train_mean
                if 'SADI' in models.keys():
                    samples_sadi_mean = (samples_sadi_mean * train_std) + train_mean
                if 'SAITS' in models.keys():
                    saits_output = (torch.tensor(saits_output, device=device) * train_std) + train_mean
                if 'MICE' in models.keys():
                    mice_output = (torch.tensor(mice_output, device=device) * train_std) + train_mean
                if 'BRITS' in models.keys():
                    brits_output = (torch.tensor(brits_output, device=device) * train_std) + train_mean
                c_target = (c_target * train_std) + train_mean  
            else:
                if 'CSDI' in models.keys():
                    samples_median_csdi = samples_median.values

            if data:
                for idx in range(samples.shape[1]):
                    if 'CSDI' in models.keys():
                        samples[0, idx] = (samples[0, idx] * train_std) + train_mean
                    if 'SADI' in models.keys():
                        samples_sadi[0, idx] = (samples_sadi[0, idx] * train_std) + train_mean
                results_data[j] = {
                    'target mask': eval_points[0, :, :].cpu().numpy(),
                    'target': c_target[0, :, :].cpu().numpy(),
                    'observed_mask': test_batch['observed_mask'][0, :, :].cpu().numpy()
                }
                if 'CSDI' in models.keys():
                        results_data[j]['csdi_median'] = samples_median_csdi[0, :, :].cpu().numpy()
                        results_data[j]['csdi_samples'] = samples[0].cpu().numpy()
                if 'SADI' in models.keys():
                        results_data[j]['sadi_mean'] = samples_sadi_mean[0, :, :].cpu().numpy()
                        results_data[j]['sadi_samples'] = samples_sadi[0].cpu().numpy()
                        results_data[j]['sadi_median'] = samples_sadi_median.values[0, :, :].cpu().numpy()

                if 'SAITS' in models.keys():
                    results_data[j]['saits'] = saits_output[0, :, :].cpu().numpy()

                if 'KNN' in models.keys():
                    results_data[j]['knn'] = knn_output[0, :, :].cpu().numpy()

                if 'MICE' in models.keys():
                    results_data[j]['mice'] = mice_output[0, :, :].cpu().numpy()
                
                if 'BRITS' in models.keys():
                    results_data[j]['brits'] = brits_output[0, :, :].cpu().numpy()
            else:
                ###### CSDI ######
                if 'CSDI' in models.keys():
                    rmse_csdi = ((samples_median_csdi - c_target) * eval_points) ** 2
                    rmse_csdi = rmse_csdi.sum().item() / eval_points.sum().item()
                    csdi_rmse_avg += rmse_csdi

                    mae_csdi = torch.abs((samples_median_csdi - c_target) * eval_points)
                    mae_csdi = mae_csdi.sum().item() / eval_points.sum().item()
                    csdi_mae_avg += mae_csdi

                    csdi_crps = calc_quantile_CRPS(c_target, samples, eval_points, 0, 1)
                    csdi_crps_avg += csdi_crps

                ###### SADI ######
                if 'SADI' in models.keys():
                    rmse_sadi = ((samples_sadi_mean - c_target) * eval_points) ** 2
                    rmse_sadi = rmse_sadi.sum().item() / eval_points.sum().item()
                    sadi_rmse_avg += rmse_sadi

                    mse_sadi_median = ((samples_sadi_median.values - c_target) * eval_points) ** 2
                    mse_sadi_median = mse_sadi_median.sum().item() / eval_points.sum().item()
                    sadi_median_avg += mse_sadi_median

                    mse_sadi_mean_med = ((samples_sadi_mean_med - c_target) * eval_points) ** 2
                    mse_sadi_mean_med = mse_sadi_mean_med.sum().item() / eval_points.sum().item()
                    sadi_mean_med_avg += mse_sadi_mean_med

                    mae_sadi = torch.abs((samples_sadi_mean - c_target) * eval_points)
                    mae_sadi = mae_sadi.sum().item() / eval_points.sum().item()
                    sadi_mae_avg += mae_sadi

                    sadi_crps = calc_quantile_CRPS(c_target, samples_sadi, eval_points, 0, 1)
                    sadi_crps_avg += sadi_crps

                ###### SAITS ######
                if 'SAITS' in models.keys():
                    rmse_saits = ((torch.tensor(saits_output, device=device)- c_target) * eval_points) ** 2
                    rmse_saits = rmse_saits.sum().item() / eval_points.sum().item()
                    saits_rmse_avg += rmse_saits
                
                    mae_saits = torch.abs((torch.tensor(saits_output, device=device)- c_target) * eval_points)
                    mae_saits = mae_saits.sum().item() / eval_points.sum().item()
                    saits_mae_avg += mae_saits

                ###### KNN ######
                if 'KNN' in models.keys():
                    rmse_knn = ((torch.tensor(knn_output, device=device)- c_target) * eval_points) ** 2
                    rmse_knn = rmse_knn.sum().item() / eval_points.sum().item()
                    knn_rmse_avg += rmse_knn

                ###### MICE ######
                if 'MICE' in models.keys():
                    rmse_mice = ((torch.tensor(mice_output, device=device) - c_target) * eval_points) ** 2
                    rmse_mice = rmse_mice.sum().item() / eval_points.sum().item()
                    mice_rmse_avg += rmse_mice

                ###### BRITS ######
                if 'BRITS' in models.keys():
                    rmse_brits = ((torch.tensor(brits_output, device=device) - c_target) * eval_points) ** 2
                    rmse_brits = rmse_brits.sum().item() / eval_points.sum().item()
                    brits_rmse_avg += rmse_brits
        if not data:
            if 'CSDI' in models.keys():
                results_trials_mse['csdi'][trial] = csdi_rmse_avg / batch_size
                results_mse['csdi'] += csdi_rmse_avg / batch_size
                results_trials_mae['csdi'][trial] = csdi_mae_avg / batch_size
                results_mae['csdi'] += csdi_mae_avg / batch_size
                results_crps['csdi_trials'][trial] = csdi_crps_avg / batch_size
                results_crps['csdi'] += csdi_crps_avg / batch_size

            if 'SADI' in models.keys():
                results_trials_mse['sadi'][trial] = sadi_rmse_avg / batch_size
                results_mse['sadi'] += sadi_rmse_avg / batch_size
                results_mse['sadi_median'] += sadi_median_avg / batch_size
                results_mse['sadi_mean_med'] += sadi_mean_med_avg / batch_size
                results_trials_mae['sadi'][trial] = sadi_mae_avg / batch_size
                results_mae['sadi'] += sadi_mae_avg / batch_size
                results_crps['sadi_trials'][trial] = sadi_crps_avg / batch_size
                results_crps['sadi'] += sadi_crps_avg / batch_size
                

            if 'SAITS' in models.keys():
                results_trials_mse['saits'][trial] = saits_rmse_avg / batch_size
                results_mse['saits'] += saits_rmse_avg / batch_size
                results_trials_mae['saits'][trial] = saits_mae_avg / batch_size
                results_mae['saits'] += saits_mae_avg / batch_size
     
            if 'KNN' in models.keys():
                results_trials_mse['knn'][trial] = knn_rmse_avg / batch_size
                results_mse['knn'] += knn_rmse_avg / batch_size
            
            if 'MICE' in models.keys():
                results_trials_mse['mice'][trial] = mice_rmse_avg / batch_size
                results_mse['mice'] += mice_rmse_avg / batch_size

            if 'BRITS' in models.keys():
                results_trials_mse['brits'][trial] = brits_rmse_avg / batch_size
                results_mse['brits'] += brits_rmse_avg / batch_size
    
    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    
    if not data:
        results_mse['csdi'] /= trials
        results_mse['sadi'] /= trials
        results_mse['sadi_median'] /= trials
        results_mse['sadi_mean_med'] /= trials
        results_mse['saits'] /= trials
        results_mse['knn'] /= trials
        results_mse['mice'] /= trials
        results_mse['brits'] /= trials

        z = 1.96
        csdi_trials = -1
        csdi_crps_ci = -1
        sadi_trials = -1
        sadi_crps_ci = -1
        saits_trials = -1
        knn_trials = -1
        mice_trials = -1
        brits_trials = -1
        if 'CSDI' in models.keys():
            csdi_trials = [results_trials_mse['csdi'][i] for i in results_trials_mse['csdi'].keys()]
            csdi_trials = (z * np.std(csdi_trials)) / math.sqrt(len(csdi_trials))
            csdi_crps_ci = [results_crps['csdi_trials'][i] for i in results_crps['csdi_trials'].keys()]
            csdi_crps_ci = (z * np.std(csdi_crps_ci)) / math.sqrt(len(csdi_crps_ci))

        if 'SADI' in models.keys():
            sadi_trials = [results_trials_mse['sadi'][i] for i in results_trials_mse['sadi'].keys()]
            sadi_trials = (z * np.std(sadi_trials)) / math.sqrt(len(sadi_trials))
            sadi_crps_ci = [results_crps['sadi_trials'][i] for i in results_crps['sadi_trials'].keys()]
            sadi_crps_ci = (z * np.std(sadi_crps_ci)) / math.sqrt(len(sadi_crps_ci))

        if "SAITS" in models.keys():
            saits_trials = [results_trials_mse['saits'][i] for i in results_trials_mse['saits'].keys()]
            saits_trials = (z * np.std(saits_trials)) / math.sqrt(len(saits_trials))

        if 'KNN' in models.keys():
            knn_trials = [results_trials_mse['knn'][i] for i in results_trials_mse['knn'].keys()]
            knn_trials = (z * np.std(knn_trials)) / math.sqrt(len(knn_trials))

        if 'MICE' in models.keys():
            mice_trials = [results_trials_mse['mice'][i] for i in results_trials_mse['mice'].keys()]
            mice_trials = (z * np.std(mice_trials)) / math.sqrt(len(mice_trials))

        if 'BRITS' in models.keys():
            brits_trials = [results_trials_mse['brits'][i] for i in results_trials_mse['brits'].keys()]
            brits_trials = (z * np.std(brits_trials)) / math.sqrt(len(brits_trials))

        print(f"MSE loss:\n\tCSDI: {results_mse['csdi']} ({csdi_trials})\n\tSADI: {results_mse['sadi']} \
              ({sadi_trials})\n\tSAITS: {results_mse['saits']} ({saits_trials})\n\tKNN: {results_mse['knn']} ({knn_trials}) \
                    \n\tMICE: {results_mse['mice']} ({mice_trials})\n\tBRITS: {results_mse['brits']} ({brits_trials})")

        results_mae['csdi'] /= trials
        results_mae['sadi'] /= trials
        results_mae['saits'] /= trials

        
        results_crps['csdi'] /= trials
        results_crps['sadi'] /= trials

        print(f"CRPS:\n\tCSDI: {results_crps['csdi']} ({csdi_crps_ci})\n\SADI: {results_crps['sadi']} ({sadi_crps_ci})")

        fp = open(f"{mse_folder}/mse-trials-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_trials_mse, fp=fp, indent=4)
        fp.close()

        fp = open(f"{mse_folder}/mae-trials-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_trials_mae, fp=fp, indent=4)
        fp.close()

        fp = open(f"{mse_folder}/mse-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_mse, fp=fp, indent=4)
        fp.close()

        fp = open(f"{mse_folder}/mae-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_mae, fp=fp, indent=4)
        fp.close()
        
        fp = open(f"{mse_folder}/crps-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_crps, fp=fp, indent=4)
        fp.close()
    else:
        fp = open(f"{mse_folder}/data-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}_pbm_{-1 if partial_bm_config is None else partial_bm_config['features']}.json", "w")
        json.dump(results_data, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()

def clip_pattern_mask(mask):
    mask = np.where(mask < 0, 0, mask)
    mask = np.where(mask > 1, 1, mask)
    return np.round(mask)
