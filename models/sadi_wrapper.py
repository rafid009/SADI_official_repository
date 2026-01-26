"""
Some parts of the code is from https://github.com/ermongroup/CSDI.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.sadi import SADI
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class SADI_base(nn.Module):

    """
    The SADI_base class is the foundational model for Self-Attention Diffusion Imputation (SADI). 
    It manages the overall process of data imputation in time-series datasets using diffusion models 
    combined with self-attention mechanisms. This class handles the configuration, forward pass, 
    and evaluation of the imputation model.

    Attributes:
        device (torch.device): The device (CPU/GPU) where the model computations will take place.
        target_dim (int): The dimension of the target data being imputed.
        target_strategy (str): The strategy used for targeting the imputation (e.g., "mix", "blackout").
        model_type (str): The type of model being used, as specified in the configuration.
        is_simple (bool): Flag indicating whether a simplified version of the model is used.
        is_fast (bool): Flag indicating whether to use a faster, less accurate imputation process.
        diffmodel (SADI): The core diffusion model used for imputation.
        num_steps (int): The number of diffusion steps in the imputation process.
        beta (numpy.ndarray): The schedule of beta values used in the diffusion process.
        alpha_hat (numpy.ndarray): The complementary alpha values used in the diffusion process.
        alpha (numpy.ndarray): The cumulative product of alpha_hat values.
        alpha_torch (torch.Tensor): The tensor version of alpha values, used in computations.
        loss_weight_p (float): The weight for the primary loss component.
        loss_weight_f (float): The weight for the feature-based loss component.
    """

    def __init__(self, target_dim, config, device, is_simple=False):
        """
        Initializes the SADI_base model with the specified configuration, setting up the diffusion process, 
        attention mechanisms, and loss function weights.

        Args:
            target_dim (int): The dimension of the target data to be imputed.
            config (dict): A dictionary containing model, diffusion, and ablation configurations.
            device (torch.device): The device (CPU/GPU) where the model computations will take place.
            is_simple (bool, optional): Flag indicating whether a simplified version of the model is used. Defaults to False.
        """
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.target_strategy = config["model"]["target_strategy"]
        self.model_type = config["model"]["type"]
        self.is_simple = is_simple
        self.is_fast = config["diffusion"]['is_fast']
        ablation_config = config['ablation']


        config_diff = config["diffusion"]
        # config_diff["side_dim"] = self.emb_total_dim

        self.diffmodel = SADI(
            diff_steps=config['diffusion']['num_steps'],
            n_layers=config['model']['n_layers'],
            d_time=config['model']['d_time'],
            d_feature=config['model']['n_feature'],
            d_model=config['model']['d_model'],
            d_inner=config['model']['d_inner'],
            n_head=config['model']['n_head'],
            d_k=config['model']['d_k'],
            d_v=config['model']['d_v'],
            dropout=config['model']['dropout'],
            diff_emb_dim=config['diffusion']['diffusion_embedding_dim'],
            diagonal_attention_mask=config['model']['diagonal_attention_mask'],
            is_simple=self.is_simple,
            ablation_config=ablation_config
        )

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            self.beta = np.cos(np.linspace(np.arccos(config_diff["beta_start"]), np.arccos(config_diff["beta_end"]), self.num_steps))
        
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        
        self.loss_weight_p = config['model']['loss_weight_p']
        self.loss_weight_f = config['model']['loss_weight_f']


    def get_randmask(self, observed_mask):
        """
        Generates a random mask based on the observed mask, simulating missing data.

        Args:
            observed_mask (torch.Tensor): The observed mask indicating where data is present.

        Returns:
            torch.Tensor: A randomly generated mask where a random proportion of the observed data is masked.
        """
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """
        Generates a historical mask based on the observed mask, optionally using a provided pattern mask.

        Args:
            observed_mask (torch.Tensor): The observed mask indicating where data is present.
            for_pattern_mask (torch.Tensor, optional): An optional mask used to guide the historical masking.

        Returns:
            torch.Tensor: A mask where a subset of the observed data is masked based on historical patterns.
        """
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask
    
    def get_bm_mask(self, observed_mask):
        """
        Generates a blackout mask by randomly setting a contiguous segment of the observed mask to zero.

        Args:
            observed_mask (torch.Tensor): The observed mask indicating where data is present.

        Returns:
            torch.Tensor: A mask where a contiguous segment of the observed data is masked (set to zero).
        """
        cond_mask = observed_mask.clone()
        for i in range(cond_mask.shape[0]):
            start = np.random.randint(0, cond_mask.shape[2] - int(cond_mask.shape[2] * 0.1))
            length = np.random.randint(int(cond_mask.shape[2] * 0.1), int(cond_mask.shape[2] * 0.2))
            cond_mask[i, :, start : (start + length - 1)] = 0.0
        return cond_mask
    
    def get_pbm_mask(self, observed_mask):
        """
        Generates a partial blackout mask, which includes both random masking and blackout segments.

        Args:
            observed_mask (torch.Tensor): The observed mask indicating where data is present.

        Returns:
            torch.Tensor: A mask where a mix of random and contiguous segments of the observed data are masked.
        """
        cond_mask = observed_mask.clone() # B, K, L
        rand_mask = self.get_randmask(observed_mask)
        for i in range(observed_mask.shape[0]):
            mask_choice = np.random.rand()
            if mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                if cond_mask.shape[1] != 1:
                    n_features = np.random.randint(1, int(np.round(cond_mask.shape[1]/2)))
                    feature_indices = np.random.choice(observed_mask.shape[1], size=n_features, replace=False)
                else:
                    feature_indices = [0]
                if observed_mask.shape[2] != 1:
                    n_time = np.random.randint(1, int(np.round(observed_mask.shape[2]/2)))
                else:
                    n_time = 1
                
                start_time = np.random.randint(0, observed_mask.shape[2] - n_time + 1)
                cond_mask[i, feature_indices, start_time:n_time] = 0.0
        return cond_mask
    

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, is_train
    ):
        """
        Calculates the validation loss by averaging the loss over all diffusion steps.

        Args:
            observed_data (torch.Tensor): The observed data tensor.
            cond_mask (torch.Tensor): The conditional mask indicating which data points are used as conditions.
            observed_mask (torch.Tensor): The observed mask indicating where data is present.
            is_train (bool): Flag indicating whether the model is in training mode.

        Returns:
            torch.Tensor: The average validation loss across all diffusion steps.
        """
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, is_train, set_t=-1
    ):
        """
        Calculates the loss for a given diffusion step during training or validation.

        Args:
            observed_data (torch.Tensor): The observed data tensor of shape (B, K, L).
            cond_mask (torch.Tensor): The conditional mask indicating which data points are used as conditions.
            observed_mask (torch.Tensor): The observed mask indicating where data is present.
            is_train (bool): Flag indicating whether the model is in training mode (1 for training, 0 for validation).
            set_t (int, optional): The specific diffusion step for which to calculate the loss. Defaults to -1.

        Returns:
            torch.Tensor: The computed loss for the specified diffusion step.
        
        Description:
            The method performs the following steps:

            1. **Diffusion Step Selection:** 
               - In validation mode, the method uses the specified diffusion step `set_t`.
               - In training mode, the diffusion step is randomly selected for each batch.

            2. **Noise Addition:** 
               - The observed data is perturbed with Gaussian noise according to the selected diffusion step. 
               - The amount of noise is controlled by the `current_alpha`, which is based on the diffusion schedule.

            3. **Input Preparation:** 
               - The noisy data, along with the original observed data and the conditional mask, is prepared 
                 as input to the diffusion model. The inputs are formatted to include both the conditional 
                 and noisy target data.

            4. **Model Prediction:** 
               - The diffusion model predicts the noise added to the data. The output includes predictions 
                 from multiple stages (if applicable), such as the first and second blocks of the model.

            5. **Loss Calculation:** 
               - The loss is computed as the squared difference between the predicted noise and the actual 
                 noise added to the observed data, weighted by the target mask (indicating missing data points). 
               - If the model is in training mode and has multiple stages, additional losses are computed for 
                 intermediate predictions, and a weighted sum of these losses is returned.

            6. **Return Loss:** 
               - The final loss is returned, averaged over the number of evaluation points.

        Example:
            loss = model.calc_loss(observed_data, cond_mask, observed_mask, is_train=1)
        """
        B, K, L = observed_data.shape
        # print(f"observed data: {observed_data}\n")
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + ((1.0 - current_alpha) ** 0.5) * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        target_mask = observed_mask - cond_mask
        num_eval = target_mask.sum()

        temp_mask = cond_mask.unsqueeze(dim=1)
        
        total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
        print(f"total_input: {total_input}\n")
        inputs = {
            'X': total_input,
            'missing_mask': total_mask
        }

        predicted_1, predicted_2, predicted_3 = self.diffmodel(inputs, t)
        print(f"predicted3: {predicted_3}\n")
        residual_3 = (noise - predicted_3) * target_mask
        
        if is_train != 0 and (predicted_1 is not None) and (predicted_2 is not None):
            pred_loss_1 = (noise - predicted_1) * target_mask
            pred_loss_2 = (noise - predicted_2) * target_mask
            pred_loss = ((pred_loss_1 ** 2).sum() + (pred_loss_2 ** 2).sum()) / 2 
            loss = (residual_3 ** 2).sum()
            loss = (self.loss_weight_f * loss + self.loss_weight_p * pred_loss) / (2 * (num_eval if num_eval > 0 else 1))
        else:
            loss = (residual_3 ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """
        Prepares the input data for the diffusion model by combining the noisy and conditional data.

        Args:
            noisy_data (torch.Tensor): The noisy version of the observed data, with noise added according to the diffusion process.
            observed_data (torch.Tensor): The original observed data before noise was added.
            cond_mask (torch.Tensor): The conditional mask indicating which data points are used as conditions.

        Returns:
            torch.Tensor: The combined input tensor, formatted with conditional and noisy data in separate channels.

        Description:
            The method concatenates the conditional observed data and the noisy target data along the channel dimension 
            to create the input tensor for the diffusion model. This tensor is used to predict the noise added during 
            the diffusion process, which is essential for the imputation of missing data.

        Example:
            diff_input = model.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        """
        print(f"observed_data: {observed_data}\n")
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def impute(self, observed_data, cond_mask, n_samples):
        """
        Generates imputed samples for the missing data points in the observed data using the diffusion model.

        Args:
            observed_data (torch.Tensor): The observed data tensor with missing values.
            cond_mask (torch.Tensor): The conditional mask indicating which data points are used as conditions.
            n_samples (int): The number of imputed samples to generate for each missing data point.

        Returns:
            torch.Tensor: A tensor containing the generated imputed samples for each missing data point.

        Description:
            The method iteratively applies the reverse diffusion process to generate multiple samples of imputed data. 
            For each diffusion step, the model refines the imputed data by predicting the noise that was added and 
            updating the imputed values accordingly. The process is repeated for a specified number of samples, 
            and the final imputed samples are returned.

        Example:
            imputed_samples = model.impute(observed_data, cond_mask, n_samples=10)
        """
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            ti = 0
            if self.is_fast:
                num_steps = 50
            else:
                num_steps = self.num_steps
            for t in range(num_steps - 1, -1, -1):

                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                temp_mask = cond_mask.unsqueeze(dim=1)
                total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)

                inputs = {
                    'X': diff_input,
                    'missing_mask': total_mask
                }

                _, _, predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device))              
                
                if self.is_fast:
                    prdicted_x0 = (current_sample - ((1 - self.alpha[t]) ** 0.5) * predicted) / (self.alpha[t] ** 0.5)
                    current_sample = (self.alpha[t-1] ** 0.5) * prdicted_x0 + ((1 - self.alpha[t-1]) ** 0.5) * predicted
                else:
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if not self.is_fast and t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    
                    current_sample += sigma * noise
                ti += 1
            current_sample = (1 - cond_mask) * current_sample + cond_mask * observed_data
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


    def forward(self, batch, is_train=1, pbm=False):
        """
        Forward pass for training or validation, processing a batch of data through the diffusion model.

        Args:
            batch (dict): A dictionary containing the input data, observed mask, ground truth mask, and other information.
            is_train (int, optional): Flag indicating whether the model is in training mode (1 for training, 0 for validation). Defaults to 1.
            pbm (bool, optional): Flag indicating whether to use Partial Blackout Masking (PBM) during training. Defaults to False.

        Returns:
            torch.Tensor: The computed loss for the batch during training or validation.

        Description:
            The `forward` method performs the following steps:

            1. **Data Processing**:
               - The method extracts the observed data, observed mask, ground truth mask, and additional information from the batch.
               - The observed data and masks are used to determine which parts of the input data are missing and need to be imputed.

            2. **Conditional Mask Generation**:
               - Depending on the `target_strategy` specified in the configuration, the method generates a conditional mask (`cond_mask`).
               - The conditional mask can be created using several strategies, including random masking, blackout masking, and partial blackout masking (PBM).

            3. **Loss Function Selection**:
               - Based on the `is_train` flag, the method selects the appropriate loss function to use:
                 - During training (`is_train=1`), the method uses the `calc_loss` function.
                 - During validation (`is_train=0`), the method uses the `calc_loss_valid` function, which averages the loss over all diffusion steps.

            4. **Loss Calculation**:
               - The selected loss function calculates the difference between the predicted imputed values and the ground truth, weighted by the missing data mask.
               - The loss is computed as the squared error between the model's predictions and the true noise added during the diffusion process.

            5. **Return Loss**:
               - The method returns the calculated loss, which is used to update the model parameters during training or to evaluate the model's performance during validation.

        Example:
            loss = model.forward(batch, is_train=1)
        """
        (
            observed_data,
            observed_mask,
            gt_mask,
            _
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy == "mix":
            cond_mask = self.get_hist_mask(
                observed_mask
            )
        elif self.target_strategy == 'blackout':
            cond_mask = self.get_bm_mask(
                observed_mask
            )
        elif self.target_strategy == 'pbm':
            if pbm:
                cond_mask = self.get_pbm_mask(observed_mask)
            else:
                cond_mask = self.get_randmask(observed_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, is_train)

    def evaluate(self, batch, n_samples):
        """
        Evaluates the model's performance on a batch of data by imputing missing values and comparing them 
        against the ground truth.

        Args:
            batch (dict): A dictionary containing the input data, observed mask, ground truth mask, and other information.
            n_samples (int): The number of imputed samples to generate for each missing data point.

        Returns:
            tuple:
                - torch.Tensor: The generated imputed samples for the missing data points.
                - torch.Tensor: The original observed data tensor with missing values.
                - torch.Tensor: The target mask indicating the positions of the missing values to be imputed.
                - torch.Tensor: The observed mask indicating where data is present.

        Description:
            The `evaluate` method performs the following steps:

            1. **Data Processing**:
               - The method extracts the observed data, observed mask, ground truth mask, and additional information from the batch.
               - The observed data and masks are used to determine which parts of the input data are missing and need to be imputed.

            2. **Mask Preparation**:
               - The method sets the conditional mask (`cond_mask`) to be the same as the ground truth mask, 
                 indicating which values are known and which are to be imputed.
               - The target mask is computed as the difference between the observed mask and the conditional mask, 
                 highlighting the missing data points that need to be imputed.

            3. **Imputation**:
               - The method generates multiple imputed samples for the missing data points by running the diffusion 
                 model in reverse, starting from randomly generated noise and refining the predictions at each diffusion step.
               - The imputed samples are generated by sampling from the model's predictions, with the process 
                 repeated `n_samples` times to produce multiple possible imputations.

            4. **Adjusting for Cut Length**:
               - For each sample in the batch, the target mask is adjusted to zero out the values corresponding to the 
                 initial cut length specified in the batch. This ensures that only the relevant portions of the 
                 time series are considered during evaluation.

            5. **Return Results**:
               - The method returns the imputed samples, the original observed data, the target mask, and the observed mask. 
                 These outputs can be used to assess the model's imputation accuracy and compare it against the ground truth.

        Example:
            samples, observed_data, target_mask, observed_mask = model.evaluate(batch, n_samples=10)
        """
        (
            observed_data,
            observed_mask,
            gt_mask,
            cut_length
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            samples = self.impute(observed_data, cond_mask, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
       
        return samples, observed_data, target_mask, observed_mask


class SADI_Agaid(SADI_base):
    """
    The SADI_Agaid class is a specialized implementation of the SADI_base model, specifically designed 
    for the AgAID dataset. It inherits the core functionalities of the SADI_base class but adapts 
    them to the unique structure and requirements of the AgAID dataset, which has a fixed target dimension.

    Attributes:
        device (torch.device): The device (CPU/GPU) where the model computations will take place.
        target_dim (int): The dimension of the target data specific to the AgAID dataset, defaulting to 21.
        is_simple (bool): Flag indicating whether a simplified version of the model is used.

    Methods:
        process_data(batch): Processes the input batch data to prepare it for the diffusion model, 
                              including permuting the dimensions to match the model's expectations.
    """
    def __init__(self, config, device, target_dim=21, is_simple=False):
        """
        Initializes the SADI_Agaid model with the specified configuration, setting up the model 
        for handling the AgAID dataset.

        Args:
            config (dict): A dictionary containing model, diffusion, and ablation configurations.
            device (torch.device): The device (CPU/GPU) where the model computations will take place.
            target_dim (int, optional): The dimension of the target data specific to the AgAID dataset. Defaults to 21.
            is_simple (bool, optional): Flag indicating whether a simplified version of the model is used. Defaults to False.
        """
        super(SADI_Agaid, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        """
        Processes the input batch data for the AgAID dataset, preparing it for the diffusion model. 
        This involves converting the data to the appropriate device, permuting dimensions, and 
        initializing the cut length tensor.

        Args:
            batch (dict): A dictionary containing the input data, observed mask, ground truth mask, 
                          and other information from the AgAID dataset.

        Returns:
            tuple:
                - torch.Tensor: The processed observed data tensor with dimensions permuted to match the model's input requirements.
                - torch.Tensor: The processed observed mask tensor with dimensions permuted to match the model's input requirements.
                - torch.Tensor: The ground truth mask tensor with dimensions permuted to match the model's input requirements.
                - torch.Tensor: The cut length tensor initialized to zeros, used in subsequent processing.
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        # observed_data_intact = batch["obs_data_intact"].to(self.device).float()
        # gt_intact = batch["gt_intact"]#.to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        return (
            observed_data,
            observed_mask,
            gt_mask,
            # for_pattern_mask,
            cut_length,
            # observed_data_intact,
            # gt_intact
        )

class SADI_Synth(SADI_base):
    def __init__(self, config, device, target_dim=6, is_simple=False):
        super(SADI_Synth, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        observed_data_intact = batch["obs_data_intact"].to(self.device).float()
        gt_intact = batch["gt_intact"]#.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            gt_mask,
            # for_pattern_mask,
            cut_length,
            # observed_data_intact,
            # gt_intact
        )

class SADI_NASCE(SADI_base):
    def __init__(self, config, device, target_dim=352, is_simple=False):
        super(SADI_NASCE, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        # observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        # observed_data_intact = batch["obs_data_intact"].to(self.device).float()
        # gt_intact = batch["gt_intact"]#.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        # for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            # observed_tp,
            gt_mask,
            # for_pattern_mask,
            cut_length,
            # None,
            # gt_intact
        )

class SADI_SWaT(SADI_base):
    def __init__(self, config, device, target_dim=51, is_simple=False):
        super(SADI_SWaT, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        # observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        # observed_data_intact = batch["obs_data_intact"].to(self.device).float()
        # gt_intact = batch["gt_intact"]#.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        # for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            # observed_tp,
            gt_mask,
            # for_pattern_mask,
            cut_length,
            # None,
            # gt_intact
        )
