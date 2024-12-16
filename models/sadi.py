import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.transformer import EncoderLayer, PositionalEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiffusionEmbedding(nn.Module):
    """
    The DiffusionEmbedding class provides a learnable embedding for diffusion steps in a diffusion model.
    It embeds each diffusion step into a higher-dimensional space, allowing the model to effectively utilize 
    information about the progression of the diffusion process.

    Attributes:
        embedding (torch.Tensor): A table containing the sinusoidal and cosinusoidal embeddings for each diffusion step.
        projection1 (nn.Linear): A linear layer that projects the embedding to the specified projection dimension.
        projection2 (nn.Linear): A second linear layer that further processes the projected embedding.
    """
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        """
        Initializes the DiffusionEmbedding class by setting up the embedding and projection layers.

        Args:
            num_steps (int): The number of diffusion steps to embed.
            embedding_dim (int, optional): The dimension of the initial embedding space. Default is 128.
            projection_dim (int, optional): The dimension to which the embedding will be projected. 
                                            If not provided, it defaults to the same value as embedding_dim.
        """
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
        """
        Performs a forward pass through the DiffusionEmbedding module, transforming the diffusion step into its embedded representation.

        Args:
            diffusion_step (torch.Tensor): A tensor representing the current diffusion step(s).

        Returns:
            torch.Tensor: The embedded representation of the diffusion step, after being processed through two projection layers and activation functions.
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        """
        Constructs the sinusoidal and cosinusoidal embedding table for the diffusion steps.

        Args:
            num_steps (int): The number of diffusion steps for which embeddings are created.
            dim (int, optional): The dimension of the sinusoidal/cosinusoidal embedding space. Default is 64.

        Returns:
            torch.Tensor: A tensor containing the sinusoidal and cosinusoidal embeddings for each diffusion step, with shape (num_steps, dim * 2).
        """
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
    

def Conv1d_with_init_saits_new(in_channels, out_channels, kernel_size, init_zero=False, dilation=1):
    """
    Creates a 1D convolutional layer with custom initialization for the weights.

    Args:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolutional kernel (filter).
        init_zero (bool, optional): If True, initializes the convolutional weights to zero. 
                                    If False, uses Kaiming normal initialization. Defaults to False.
        dilation (int, optional): Spacing between kernel elements (dilation rate). Defaults to 1.

    Returns:
        nn.Conv1d: A 1D convolutional layer with the specified initialization.

    Description:
        The function creates a 1D convolutional layer (`nn.Conv1d`) with the specified number of input and output channels,
        kernel size, and dilation rate. The padding is calculated based on the kernel size and dilation to ensure that 
        the output size is the same as the input size (if stride is 1).

        The function allows for two types of weight initialization:
        - If `init_zero` is True, the weights are initialized to zero.
        - If `init_zero` is False, Kaiming normal initialization is applied to the weights.

    Example:
        layer = Conv1d_with_init_saits_new(in_channels=16, out_channels=32, kernel_size=3, init_zero=True)
    """
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)

    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer


class GTA(nn.Module):
    """
    The GTA (Gated Temporal Attention) class is a neural network module designed for time-series data processing, 
    specifically to handle the temporal and feature dimensions in a diffusion model. It incorporates multiple 
    layers of attention and convolutional operations, allowing the model to effectively capture complex dependencies 
    across time steps and features.

    Attributes:
        enc_layer_1 (EncoderLayer): The first encoder layer used for initial attention over time and features.
        enc_layer_2 (EncoderLayer): The second encoder layer used for further attention and processing.
        diffusion_projection (nn.Linear): A linear layer for projecting diffusion embeddings into the channel space.
        init_proj (nn.Conv1d): A convolutional layer for initial projection of the input noise.
        conv_layer (nn.Conv1d): A convolutional layer for transforming the projected input.
        cond_proj (nn.Conv1d): A convolutional layer for initial projection of the conditioning data.
        conv_cond (nn.Conv1d): A convolutional layer for transforming the projected conditioning data.
        res_proj (nn.Conv1d): A convolutional layer for projecting the output back to the original feature space.
        skip_proj (nn.Conv1d): A convolutional layer for producing the skip connection.
    """
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        """
        Initializes the GTA module by setting up the layers and projections required for processing
        the time-series data and diffusion embeddings.

        Args:
            channels (int): The number of channels used in the convolutional layers.
            d_time (int): The dimension of the time axis in the attention mechanism.
            actual_d_feature (int): The actual dimension of features in the input data.
            d_model (int): The dimension of the input model.
            d_inner (int): The inner dimension in the encoder layers.
            n_head (int): The number of attention heads in the encoder layers.
            d_k (int): The dimension of the key in the attention mechanism.
            d_v (int): The dimension of the value in the attention mechanism.
            dropout (float): The dropout rate applied to the attention mechanism.
            diffusion_embedding_dim (int, optional): The dimension of the diffusion embedding. Defaults to 128.
            diagonal_attention_mask (bool, optional): Whether to use a diagonal mask in the attention mechanism. Defaults to True.
        """
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
        """
        Forward pass for the GTA module. This method processes the input noise (x), conditioning data (cond),
        and diffusion embeddings (diffusion_emb) through multiple layers of attention and convolution.

        Args:
            x (torch.Tensor): The input noise tensor of shape (B, L, K), where B is the batch size, 
                              L is the sequence length (time dimension), and K is the number of features.
            cond (torch.Tensor): The conditioning data tensor of shape (B, L, K).
            diffusion_emb (torch.Tensor): The diffusion embedding tensor of shape (B, embedding_dim).

        Returns:
            tuple:
                - torch.Tensor: The processed output tensor with residual connections.
                - torch.Tensor: The skip connection tensor.
                - torch.Tensor: The attention weights averaged over the two encoder layers.
        """
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
    """
    The SADI (Self-Attention Diffusion Imputation) class is a neural network model designed for imputing missing data 
    in time-series datasets. The model leverages a combination of temporal and feature-based attention mechanisms, 
    as well as a diffusion process to generate accurate imputations. It is highly configurable, supporting various 
    architectural choices and ablation settings.

    Attributes:
        n_layers (int): The number of layers in the first and second blocks of the model.
        is_simple (bool): A flag indicating whether a simplified version of the model is used.
        d_feature (int): The dimension of the feature space in the input data.
        ablation_config (dict): Configuration dictionary for various ablation settings.
        d_time (int): The dimension of the time axis in the attention mechanism.
        n_head (int): The number of attention heads in the encoder layers.
        layer_stack_for_first_block (nn.ModuleList): A stack of `GTA` layers used in the first block.
        layer_stack_for_second_block (nn.ModuleList, optional): A stack of `GTA` layers used in the second block (if enabled).
        diffusion_embedding (DiffusionEmbedding): Embedding for the diffusion process.
        dropout (nn.Dropout): Dropout layer to apply regularization.
        position_enc_cond (PositionalEncoding): Positional encoding for the conditioning data.
        position_enc_noise (PositionalEncoding): Positional encoding for the noise data.
        embedding_1 (nn.Linear): Linear layer for embedding the input data for the first block.
        embedding_cond (nn.Linear): Linear layer for embedding the conditioning data.
        reduce_skip_z (nn.Linear): Linear layer to reduce the dimension of the skip connections.
        weight_combine (nn.Linear, optional): Linear layer to combine weights from the first and second blocks.
        mask_conv (nn.Conv1d): Convolutional layer used in the Feature Dependency Encoder (FDE).
        layer_stack_for_feature_weights (nn.ModuleList): A stack of `EncoderLayer` layers used in the FDE.
        fde_pos_enc (PositionalEncoding, optional): Positional encoding applied to the FDE output.
        fde_time_pos_enc (PositionalEncoding, optional): Time-based positional encoding applied to the FDE output.
    """
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=False, is_simple=False, ablation_config=None):
        """
        Initializes the SADI model with the specified configuration, setting up the necessary layers 
        and attention mechanisms for processing time-series data and imputing missing values.

        Args:
            diff_steps (int): The number of diffusion steps in the model.
            diff_emb_dim (int): The dimension of the diffusion embedding.
            n_layers (int): The number of layers in the first and second blocks of the model.
            d_time (int): The dimension of the time axis in the attention mechanism.
            d_feature (int): The dimension of the feature space in the input data.
            d_model (int): The dimension of the input model.
            d_inner (int): The inner dimension in the encoder layers.
            n_head (int): The number of attention heads in the encoder layers.
            d_k (int): The dimension of the key in the attention mechanism.
            d_v (int): The dimension of the value in the attention mechanism.
            dropout (float): The dropout rate applied to the attention mechanism.
            diagonal_attention_mask (bool, optional): Whether to use a diagonal mask in the attention mechanism. Defaults to False.
            is_simple (bool, optional): Flag indicating whether a simplified version of the model is used. Defaults to False.
            ablation_config (dict, optional): Configuration dictionary for various ablation settings. Defaults to None.
        """
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
                                self.ablation_config['fde-diagonal'], choice='fde-conv-multi', dilation=(i+1))
                    for i in range(self.ablation_config['fde-layers'])
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
        """
        Forward pass for the SADI model. Processes the input data through various attention and diffusion mechanisms
        to impute missing values in the time-series data.

        Args:
            inputs (dict): A dictionary containing the input time-series data and the missing mask. 
                           - 'X': The input time-series data of shape (B, 2, L, K), where B is the batch size,
                             L is the sequence length, and K is the number of features.
                           - 'missing_mask': The mask indicating missing values in the input data.
            diffusion_step (torch.Tensor): The diffusion step tensor representing the current step in the diffusion process.

        Returns:
            tuple:
                - torch.Tensor: The output tensor from the first block of the model, potentially including residual and skip connections.
                - torch.Tensor: The output tensor from the second block of the model (if applicable), representing further refined imputation results.
                - torch.Tensor: The final output tensor after combining results from both blocks (if applicable) or from the first block alone.

        Description:
            The forward method performs the following operations:

            1. **Input Preparation**: The input time-series data and the corresponding missing mask are processed and 
               transposed to match the expected input format for the convolutional and attention layers.

            2. **Feature Dependency Encoder (FDE)**: If enabled by the ablation configuration, the model first processes 
               the input through a Feature Dependency Encoder to capture global correlations between features over time. 
               The output of this step is a feature attention/dependency matrix, which is used to enhance the imputation.

            3. **First Block of GTA Layers**: The input data is then passed through a stack of `GTA` (Gated Temporal Attention) 
               layers in the first block, where temporal and feature-based attention mechanisms are applied. 
               The output includes skip connections which are aggregated across all layers in the block.

            4. **Second Block of GTA Layers**: If the second block is enabled (based on the ablation configuration), the 
               output of the first block is further processed through a second stack of `GTA` layers. 
               The results from this block are combined with those from the first block to produce refined imputations.

            5. **Weight Combination**: If the weight combination mechanism is enabled, the model combines the results 
               from the first and second blocks using a learned set of weights, which are determined by the attention 
               weights and the missing mask.

            6. **Output**: Finally, the method returns the outputs from the first block, the second block (if applicable), 
               and the combined result. The output represents the imputed time-series data with missing values filled in.

        Example:
            skips_tilde_1, skips_tilde_2, skips_tilde_3 = model.forward(inputs={'X': time_series_data, 'missing_mask': mask}, diffusion_step=step)
        """
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

