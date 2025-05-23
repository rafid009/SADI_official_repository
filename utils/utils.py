import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from json import JSONEncoder
import os
from datasets.dataset_synth import get_testloader_synth
from datasets.dataset_agaid import get_testloader_agaid
from datasets.dataset_nasce import get_testloader_nasce
import matplotlib.pyplot as plt
import math
import sys
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=torch.inf)


def get_num_params(model):
    """
    Calculate the total number of trainable parameters in a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which to calculate the number of parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    
    Example:
        model = MyModel()
        total_params = get_num_params(model)
        print(f"Total trainable parameters: {total_params}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    filename="",
    is_saits=False,
    data_type="",
    pbm_start=-1
):
    """
    Train a PyTorch model using a specified training configuration and dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        config (dict): Configuration dictionary containing training parameters such as the number of epochs and learning rate.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader, optional): DataLoader for the validation dataset. Defaults to None.
        valid_epoch_interval (int, optional): Interval (in epochs) at which the model is validated. Defaults to 5.
        foldername (str, optional): Directory where model checkpoints and plots will be saved. Defaults to "".
        filename (str, optional): Name of the file to save the trained model. Defaults to "".
        is_saits (bool, optional): Flag to indicate if the model is a SAITS variant, influencing the learning rate scheduler. Defaults to False.
        data_type (str, optional): Type of dataset used, which may influence specific training behaviors. Defaults to "".
        pbm_start (float, optional): The proportion of epochs after which partial blackout mode (PBM) is activated. If -1, PBM is not used. Defaults to -1.

    Returns:
        None

    Description:
    The `train` function trains a PyTorch model using a specified training configuration and dataset. It supports different learning rate scheduling strategies based on the model type and dataset. The training process includes logging of training and validation losses, saving model checkpoints, and generating loss plots.

    - The function initializes an optimizer (`Adam`) and a learning rate scheduler (`MultiStepLR`), with the schedule depending on the dataset and model type.
    - It iterates through the specified number of epochs, computing and backpropagating the loss for each batch in the training dataset.
    - If partial blackout mode (PBM) is activated (controlled by `pbm_start`), the model is trained with this mode after the specified proportion of epochs.
    - The model is periodically validated based on the `valid_epoch_interval` parameter, and the best model is saved to the specified directory.
    - After training is complete, the function generates a plot of the training loss over epochs and saves it to the specified directory.

    The function is highly customizable, allowing for various training strategies, including random missing scenarios and partial blackout training, making it suitable for different types of time-series imputation tasks.
    """
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + f"/{filename if len(filename) != 0 else 'model_csdi.pth'}"

    # p0 = int(0.6 * config["epochs"])
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    p3 = int(0.8 * config["epochs"])
    # p4 = int(0.7 * config["epochs"])
    p5 = int(0.6 * config["epochs"])
    # exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if is_saits:
        if data_type == 'agaid':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        elif data_type == 'pm25':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2], gamma=0.1
            )
        # pa
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
        )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=1000, T_mult=1, eta_min=1.0e-7
    #     )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)
    losses_y = []
    best_valid_loss = 1e10
    epoch_pbm_starts = int(config['epochs'] * pbm_start)
    model.train()
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        # if epoch_no == 1000:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        # if epoch_no > 1000 and epoch_no % 500 == 0:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # print(f"train data: {train_batch}")
                if pbm_start != -1 and epoch_no > epoch_pbm_starts:
                    loss = model(train_batch, pbm=True)
                else:
                    loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                # lr_scheduler.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            losses_y.append(avg_loss / batch_no)
            # exp_scheduler.step()
            # metric = avg_loss / batch_no
            if is_saits:
                # if data_type != 'pm25' and data_type != 'synth_v2' and data_type != 'synth_v3':
                #     lr_scheduler.step()
                # pass
                # if data_type == 'electricity':
                #     pass
                # else:
                if data_type != 'nasce' and data_type != 'pm25' and data_type != 'electricity' and data_type != 'synth_v8':
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
                # pass
            
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
                # print(
                #     "\n avg loss is now ",
                #     avg_loss_valid / batch_no,
                #     "at",
                #     epoch_no,
                # )

    if filename != "":
        torch.save(model.state_dict(), output_path)
    # if filename != "":
    #     torch.save(model.state_dict(), filename)
    x = np.arange(len(losses_y))
    folder = f"{foldername}/plots"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.figure(figsize=(16,9))
    plt.plot(x,losses_y, label='Training Loss')
    plt.title(f"Training Losses for {data_type}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{folder}/{data_type}_{'pbm' if pbm_start != -1 else 'random'}.png", dpi=300)
    plt.tight_layout()
    plt.close()


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    """
    Calculate the quantile loss for a given quantile `q`.

    The quantile loss is used to evaluate the accuracy of probabilistic forecasts, particularly in the context of the CRPS calculation.
    It computes the weighted absolute difference between the forecasted values and the actual target values, adjusted for the specified quantile.

    Args:
        target (torch.Tensor): The ground truth values.
        forecast (torch.Tensor): The predicted forecast values.
        q (float): The quantile to be used in the loss calculation (e.g., 0.05, 0.5).
        eval_points (torch.Tensor): A mask or indicator tensor specifying the points in the target to be evaluated.

    Returns:
        float: The computed quantile loss.
    """
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    """
    Calculate the denominator for the CRPS calculation.

    This denominator is the sum of the absolute values of the target, masked by `eval_points`.
    It serves as a normalization factor in the CRPS calculation to ensure the score is properly scaled.

    Args:
        target (torch.Tensor): The ground truth values.
        eval_points (torch.Tensor): A mask or indicator tensor specifying the points in the target to be evaluated.

    Returns:
        torch.Tensor: The calculated denominator for the CRPS.
    """
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using quantile losses.

    The CRPS is a measure of the accuracy of probabilistic forecasts. This function calculates the CRPS by averaging
    quantile losses across a range of quantiles (0.05 to 0.95). The target and forecast values are first rescaled
    using the provided mean and standard scalers.

    Args:
        target (torch.Tensor): The ground truth values.
        forecast (torch.Tensor): The predicted forecast values.
        eval_points (torch.Tensor): A mask or indicator tensor specifying the points in the target to be evaluated.
        mean_scaler (torch.Tensor): The mean scaler used to rescale the data.
        scaler (torch.Tensor): The standard scaler used to rescale the data.

    Returns:
        float: The calculated CRPS score.
    """
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
    """
    Evaluate the performance of various imputation models across different datasets and scenarios.

    Args:
        models (dict): A dictionary containing the models to evaluate, with keys as model names (e.g., 'CSDI', 'SADI') and values as the respective model objects.
        mse_folder (str): Path to the directory where the evaluation results (MSE, MAE, CRPS) will be saved.
        dataset_name (str, optional): The name of the dataset to use for evaluation. Determines the data loader and configuration. Defaults to ''.
        batch_size (int, optional): Batch size for evaluation. Defaults to 16.
        trials (int, optional): Number of trials to run for evaluation. Each trial may use different random seeds or data lengths. Defaults to 3.
        length (int or tuple, optional): Length of the sequence to be used in evaluation. Can be a single integer or a tuple specifying a range. Defaults to -1.
        random_trial (bool, optional): If True, evaluates on random missing scenario. Defaults to False.
        forecasting (bool, optional): If True, evaluates models in a forecasting scenario instead of imputation. Defaults to False.
        missing_ratio (float, optional): The ratio of missing data in the dataset. Defaults to 0.01.
        test_indices (list, optional): Specific indices to evaluate within the dataset. Defaults to None.
        data (bool, optional): If True, saves the imputed data for further analysis. Defaults to False.
        noise (bool, optional): If True, introduces noise into the dataset for evaluation. Defaults to False.
        filename (str, optional): Filename for loading specific datasets or saving results. Defaults to None.
        is_yearly (bool, optional): Flag indicating if the data is yearly (used for some datasets). Defaults to True.
        n_steps (int, optional): Number of steps in the time series, e.g., days in a year for daily data. Defaults to 366.
        pattern (dict, optional): Specifies the pattern of missing data, if applicable. Defaults to None.
        mean (float or ndarray, optional): Mean value(s) for data normalization, if applicable. Defaults to None.
        std (float or ndarray, optional): Standard deviation value(s) for data normalization, if applicable. Defaults to None.
        partial_bm_config (dict, optional): Configuration for partial blackout mode (PBM), defining features or intervals for PBM. Defaults to None.
        exclude_features (list, optional): Features to exclude from evaluation, if any. Defaults to None.
        unnormalize (bool, optional): If True, unnormalizes the data back to its original scale after imputation. Defaults to False.

    Returns:
        None

    Description:
    The `evaluate_imputation_all` function evaluates the performance of various time-series imputation models (e.g., CSDI, SADI, SAITS, KNN, MICE, BRITS) across different datasets and scenarios, including forecasting and imputation under partial blackout conditions. The function supports multiple trials to account for variability and randomness in data handling.

    The evaluation metrics include Mean Squared Error (MSE), Mean Absolute Error (MAE), and Continuous Ranked Probability Score (CRPS). The function calculates these metrics over the specified trials and saves the results to the provided directory (`mse_folder`). Additionally, it supports saving the imputed data for further analysis.

    The function is highly configurable, allowing for different datasets, data lengths, randomization, and even noisy data to be tested. It ensures that the models are evaluated fairly by unnormalizing the data when required and applying various data patterns (e.g., missing data scenarios).

    Finally, the results of the evaluations are saved in JSON format for each trial, making it easy to analyze and compare model performance.
    """
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
    s = 10
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
        elif dataset_name == 'nasce':
            test_loader = get_testloader_nasce(filename[0], filename[1], n_steps=n_steps, batch_size=batch_size, missing_ratio=missing_ratio, seed=(s + trial), length=length, forecasting=forecasting, random_trial=random_trial, pattern=pattern, partial_bm_config=partial_bm_config)
        else:
            test_loader = get_testloader_agaid(seed=(s + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecastig=forecasting, batch_size=batch_size, test_idx=test_indices, mean=mean, std=std, pattern=pattern, partial_bm_config=partial_bm_config)

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
        total_batch = 0
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for j, test_batch in enumerate(it, start=1):
                if 'CSDI' in models.keys():
                    output = models['CSDI'].evaluate(test_batch, nsample)
                    samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                    samples_median = samples.median(dim=1)
                
                if 'SADI' in models.keys():
                    if data:
                        start = time.time()

                    output_sadi = models['SADI'].evaluate(test_batch, nsample)
                    if data:
                        end = time.time()
                        print(f"time: {(end-start)/1000}s")
                    if 'CSDI' not in models.keys():
                        samples_sadi, c_target, eval_points, observed_points = output_sadi
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
                    elif dataset_name == 'awn':
                        folder = "./data/AWN/singles"
                        train_mean = np.load(f"{folder}/{filename.split('.')[0]}_mean.npy")
                        train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
                        train_std = np.load(f"{folder}/{filename.split('.')[0]}_std.npy")
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
                    if 'CSDI' in models.keys():
                        s = samples
                    else:
                        s = samples_sadi
                    for idx in range(s.shape[1]):
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
                total_batch += 1
                
        if not data:
            if 'CSDI' in models.keys():
                results_trials_mse['csdi'][trial] = csdi_rmse_avg / total_batch
                results_mse['csdi'] += csdi_rmse_avg / total_batch
                results_trials_mae['csdi'][trial] = csdi_mae_avg / total_batch
                results_mae['csdi'] += csdi_mae_avg / total_batch
                results_crps['csdi_trials'][trial] = csdi_crps_avg / total_batch
                results_crps['csdi'] += csdi_crps_avg / total_batch

            if 'SADI' in models.keys():
                results_trials_mse['sadi'][trial] = sadi_rmse_avg / total_batch
                results_mse['sadi'] += sadi_rmse_avg / total_batch
                results_mse['sadi_median'] += sadi_median_avg / total_batch
                results_mse['sadi_mean_med'] += sadi_mean_med_avg / total_batch
                results_trials_mae['sadi'][trial] = sadi_mae_avg / total_batch
                results_mae['sadi'] += sadi_mae_avg / total_batch
                results_crps['sadi_trials'][trial] = sadi_crps_avg / total_batch
                results_crps['sadi'] += sadi_crps_avg / total_batch
                

            if 'SAITS' in models.keys():
                results_trials_mse['saits'][trial] = saits_rmse_avg / total_batch
                results_mse['saits'] += saits_rmse_avg / batch_size
                results_trials_mae['saits'][trial] = saits_mae_avg / batch_size
                results_mae['saits'] += saits_mae_avg / total_batch
     
            if 'KNN' in models.keys():
                results_trials_mse['knn'][trial] = knn_rmse_avg / total_batch
                results_mse['knn'] += knn_rmse_avg / total_batch
            
            if 'MICE' in models.keys():
                results_trials_mse['mice'][trial] = mice_rmse_avg / total_batch
                results_mse['mice'] += mice_rmse_avg / total_batch

            if 'BRITS' in models.keys():
                results_trials_mse['brits'][trial] = brits_rmse_avg / total_batch
                results_mse['brits'] += brits_rmse_avg / total_batch
    
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
