config = {
    'train': {
        'epochs': 5000,
        'batch_size': 16 ,
        'lr': 1.0e-3
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.7,
        'num_steps': 50,
        'schedule': "quad",
        'is_fast': False
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'SAITS',
        'n_layers': 4,
        'loss_weight_p': 1,
        'loss_weight_f': 1,
        'd_time': 252,
        'n_feature': 21, #len(features),
        'd_model': 128,
        'd_inner': 128,
        'n_head': 8,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    },

    "ablation": {
        "fde-choice": "fde-conv-multi",
        "fde-layers": 4,
        "is_fde": True, # FDE
        'weight_combine': True, # wt comb
        'fde-no-mask': False,
        'fde-diagonal': True,
        'is_fde_2nd': False,
        'fde-pos-enc': True,
        'fde-time-pos-enc': False,
        'reduce-type': 'linear',
        'embed-type': 'linear',
        'is_2nd_block': True, # 2nd block
        'is-not-residual': False,
        'res-block-mask': False, 
        'is-fde-loop': False,
        'skip-connect-no-res-layer': False,
        'enc-dec': False,
        'is_stable': True,
        'is_first': True,
        'is_dual': False,
        'res_connect_blocks': False,
        'is-fde-linear': False
    }
}