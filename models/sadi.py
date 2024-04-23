import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.transformer import EncoderLayer, PositionalEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d_with_init_saits(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    

def Conv1d_with_init_saits_new(in_channels, out_channels, kernel_size, init_zero=False, dilation=1):
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
    # layer = nn.utils.weight_norm(layer)
    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer


class GTA(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()

        # combi 2
        self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        
        self.enc_layer_2 = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.cond_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_cond = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)


        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)


    def forward(self, x, cond, diffusion_emb):
        # x Noise
        # L -> time
        # K -> feature
        B, L, K = x.shape

        x_proj = torch.transpose(x, 1, 2) # (B, K, L)
        x_proj = self.init_proj(x_proj)

        cond = torch.transpose(cond, 1, 2) # (B, K, L)
        cond = self.cond_proj(cond)
        

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_proj + diff_proj #+ cond

        # attn1
        y = torch.transpose(y, 1, 2) # (B, L, channels)
        y, attn_weights_1 = self.enc_layer_1(y)
        y = torch.transpose(y, 1, 2)


        y = self.conv_layer(y)
        c_y = self.conv_cond(cond)
        y = y + c_y


        y = torch.transpose(y, 1, 2) # (B, L, 2*channels)
        y, attn_weights_2 = self.enc_layer_2(y)
        y = torch.transpose(y, 1, 2)
 

        y1, y2 = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, channels, L)

        residual = self.res_proj(out) # (B, K, L)
        residual = torch.transpose(residual, 1, 2) # (B, L, K)

        skip = self.skip_proj(out) # (B, K, L)
        skip = torch.transpose(skip, 1, 2) # (B, L, K)


        attn_weights = (attn_weights_1 + attn_weights_2) / 2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights



class SADI(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False, ablation_config=None):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        channels = d_model #int(d_model / 2)
        self.ablation_config = ablation_config
        self.d_time = d_time
        self.n_head = n_head
        
        self.layer_stack_for_first_block = nn.ModuleList([
            GTA(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        if self.ablation_config['is_2nd_block']:
            self.layer_stack_for_second_block = nn.ModuleList([
                GTA(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])

            self.embedding_2 = nn.Linear(actual_d_feature, d_model)
            self.reduce_dim_beta = nn.Linear(d_model, d_feature)
            self.reduce_dim_z = nn.Linear(d_model, d_feature)

        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)

        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.embedding_cond = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        
        if self.ablation_config['weight_combine']:
            self.weight_combine = nn.Linear(d_feature + d_time, d_feature)
        
        
        if self.ablation_config['fde-choice'] == 'fde-conv-single':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, 1, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'], choice='fde-conv-single')
                for _ in range(self.ablation_config['fde-layers'])
            ])
        elif self.ablation_config['fde-choice'] == 'fde-conv-multi':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            if not self.ablation_config['is-fde-linear']:
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                self.ablation_config['fde-diagonal'], choice='fde-conv-multi', dilation=1)
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            else:
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, 64, 64, dropout, 0,
                                self.ablation_config['fde-diagonal'], choice='fde-conv-multi', is_linear=True, dilation=1)
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            if self.ablation_config['fde-pos-enc']:
                self.fde_pos_enc = PositionalEncoding(d_time, n_position=d_feature)

            if self.ablation_config['fde-time-pos-enc']:
                self.fde_time_pos_enc = PositionalEncoding(d_feature, n_position=d_time)
        else:
            self.mask_conv = Conv1d_with_init_saits_new(2, 1, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'])
                for _ in range(self.ablation_config['fde-layers'])
            ])

    # ds3
    def forward(self, inputs, diffusion_step):
        X, masks = inputs['X'], inputs['missing_mask']
        masks[:,1,:,:] = masks[:,0,:,:]
        # B, L, K -> B=batch, L=time, K=feature
        X = torch.transpose(X, 2, 3)
        masks = torch.transpose(masks, 2, 3)
        # Feature Dependency Encoder (FDE): We are trying to get a global feature time-series cross-correlation
        # between features. Each feature's time-series will get global aggregated information from other features'
        # time-series. We also get a feature attention/dependency matrix (feature attention weights) from it.
        if self.ablation_config['is_fde'] and self.ablation_config['is_first']:
            cond_X = X[:,0,:,:] + X[:,1,:,:] # (B, L, K)
            shp = cond_X.shape
            if not self.ablation_config['fde-no-mask']:
                # In one branch, we do not apply the missing mask to the inputs of FDE
                # and in the other we stack the mask with the input time-series for each feature
                # and embed them together to get a masked informed time-series data for each feature.
                cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                # print(f"cond before mask: {cond_X.shape}")
                cond_X = self.mask_conv(cond_X) # (B, K, L)
                # print(f"cond before posenc: {cond_X.shape}")
                if self.ablation_config['fde-pos-enc']:
                    cond_X = self.fde_pos_enc(cond_X) # (B, K, L)

                if self.ablation_config['fde-time-pos-enc']:
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, L, K)
                    cond_X = self.fde_time_pos_enc(cond_X) # (B, L, K)
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)
            else:
                cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

            for feat_enc_layer in self.layer_stack_for_feature_weights:
                cond_X, _ = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

            cond_X = torch.transpose(cond_X, 1, 2) # (B, L, K)
        else:
            cond_X = X[:,1,:,:]
        
        input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)

        noise = input_X_for_first
        cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
        cond = self.embedding_cond(cond)

        diff_emb = self.diffusion_embedding(diffusion_step)
        pos_cond = self.position_enc_cond(cond)
        
        enc_output = self.dropout(self.position_enc_noise(noise))
        skips_tilde_1 = torch.zeros_like(enc_output)

        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
            skips_tilde_1 += skip

        skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)

        if self.ablation_config['is_2nd_block']:
            X_tilde_1 = self.reduce_dim_z(enc_output)
            X_tilde_1 = X_tilde_1 + skips_tilde_1 + X[:, 1, :, :]

            input_X_for_second = torch.cat([X_tilde_1, masks[:,1,:,:]], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            noise = input_X_for_second

            enc_output = self.position_enc_noise(noise)
            skips_tilde_2 = torch.zeros_like(enc_output)

            for encoder_layer in self.layer_stack_for_second_block:
                enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_2 += skip

            skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
            skips_tilde_2 = self.reduce_dim_beta(skips_tilde_2) 

            if self.ablation_config['weight_combine']:
                attn_weights = attn_weights.squeeze(dim=1)  
                if len(attn_weights.shape) == 4:
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)
                
                combining_weights = torch.sigmoid(
                    self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
                )

                skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
            else:
                skips_tilde_3 = skips_tilde_2
        if self.ablation_config['is_2nd_block']:
            if self.ablation_config['weight_combine']:
                skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
                skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
                skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
            else:
                skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
                skips_tilde_1 = None
                skips_tilde_2 = None
        else:
            skips_tilde_3 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_1 = None
            skips_tilde_2 = None

        return skips_tilde_1, skips_tilde_2, skips_tilde_3

