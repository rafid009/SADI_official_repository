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
    def __init__(self, target_dim, config, device, is_simple=False):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.target_strategy = config["model"]["target_strategy"]
        self.model_type = config["model"]["type"]
        self.is_simple = is_simple
        self.is_fast = config["diffusion"]['is_fast']
        ablation_config = config['ablation']


        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

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
        cond_mask = observed_mask.clone()
        for i in range(cond_mask.shape[0]):
            start = np.random.randint(0, cond_mask.shape[2] - int(cond_mask.shape[2] * 0.1))
            length = np.random.randint(int(cond_mask.shape[2] * 0.1), int(cond_mask.shape[2] * 0.2))
            cond_mask[i, :, start : (start + length - 1)] = 0.0
        return cond_mask
    

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, is_train
    ):
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
        B, K, L = observed_data.shape

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
        inputs = {
            'X': total_input,
            'missing_mask': total_mask
        }

        predicted_1, predicted_2, predicted_3 = self.diffmodel(inputs, t)
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
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def impute(self, observed_data, cond_mask, n_samples):
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


    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            gt_mask,
            for_pattern_mask,
            _, _, _
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy == "mix":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        elif self.target_strategy == 'blackout':
            cond_mask = self.get_bm_mask(
                observed_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            gt_mask,
            _,
            cut_length,
            obs_data_inact,
            gt_intact
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            samples = self.impute(observed_data, cond_mask, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
       
        return samples, observed_data, target_mask, observed_mask, obs_data_inact, gt_intact


class SADI_Agaid(SADI_base):
    def __init__(self, config, device, target_dim=21, is_simple=False):
        super(SADI_Agaid, self).__init__(target_dim, config, device, is_simple=is_simple)

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
            for_pattern_mask,
            cut_length,
            observed_data_intact,
            gt_intact
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
            for_pattern_mask,
            cut_length,
            observed_data_intact,
            gt_intact
        )
    

