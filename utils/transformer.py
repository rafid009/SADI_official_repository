"""
PyTorch Transformer model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

def Conv1d_with_init(in_channels, out_channels, kernel_size=1, bias=True, init_zero=False, dilation=1):
    
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, dilation=dilation)

    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer

class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout, choice='linear', d_channel=-1, is_linear=False, dilation=1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.choice = choice
        self.d_model = d_model
        self.is_linear = is_linear
        if self.choice == 'fde-conv-single' or self.choice == 'fde-conv-multi':
            if not self.is_linear:
                self.w_qs = Conv1d_with_init(d_channel, d_channel, kernel_size=3, bias=False, dilation=dilation)
                self.w_ks = Conv1d_with_init(d_channel, d_channel, kernel_size=3, bias=False, dilation=dilation)
                self.w_vs = Conv1d_with_init(d_channel, d_channel, kernel_size=3, bias=False, dilation=dilation)
                if self.choice == 'fde-conv-multi':
                    self.w_q_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
                    self.w_k_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
                    self.w_v_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
            else:
                self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
                self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
                self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if self.choice == 'fde-conv-multi':
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
            
            if not self.is_linear:
                q = self.w_qs(q).view(sz_b * len_q, 1, d_k)
                k = self.w_ks(k).view(sz_b * len_k, 1, d_k)
                v = self.w_vs(v).view(sz_b * len_v, 1, d_v)
                # q = self.w_qs(q).view(sz_b, len_q, d_k)
                # k = self.w_ks(k).view(sz_b, len_k, d_k)
                # v = self.w_vs(v).view(sz_b, len_v, d_v)

                q = self.w_q_head(q).view(sz_b, len_q, n_head, d_k)
                k = self.w_k_head(k).view(sz_b, len_k, n_head, d_k)
                v = self.w_v_head(v).view(sz_b, len_v, n_head, d_v)
            else:
                q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
                k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
                v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

            # q = q.permute(0, 2, 1, 3)
            # k = k.permute(0, 2, 1, 3)
            # v = v.permute(0, 2, 1, 3)
        else:
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)
        # print(f"v after attn: {v.shape}")
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        # print(f"v after attn+fc: {v.shape}")
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1,
                 diagonal_attention_mask=False, choice='linear', is_ffn=True, is_linear=False, dilation=1):
        super().__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout, choice=choice, d_channel=d_time, is_linear=is_linear, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.is_ffn = is_ffn
        if self.is_ffn:
            self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        else:
            self.pos_ffn = None

    def forward(self, enc_input, mask_time=None):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(enc_input.device)
 
        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        # print(f"enc_output: {enc_input.shape}")
        enc_output += residual

        if self.is_ffn:
            enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()