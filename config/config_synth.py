config = {
    'n_layers': 4,
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
    },
    'name': 'skip_fde_1st_mask_pos_enc_loss_p_bm'
}