import os
import numpy as np
import random
import torch
from models import iTransformer
from config import MockPath

def invoke(inputs):
    
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'MS',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 24,
        'enc_in': 37,
        'd_model': 64,
        'n_heads': 1,
        'e_layers': 1,
        'd_ff': 64,
        'dropout': 0.01,
        'activation': 'gelu',
        'output_attention': False
    }
    
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)
    
    test_data_root_path = inputs
    for k in range(4):
        for i in range(2):
            if i == 0:
                data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
            else: 
                data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            N, L, S, _ = data.shape # 72, 168, 60
            

            cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy")) 
            
            repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)
            C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
            covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
            data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
            C = C1 + 1
            data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
            data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
            data = data.reshape(N * S, L, C)
            data = torch.tensor(data).float().cuda() # (N * S, L, C)

            model = iTransformer.Model(args).cuda()
            
            if i == 0:
                model.load_state_dict(torch.load(f"./checkpoints/v1_iTransformer_Meteorology_ftMS_sl168_ll1_pl24_dm64_nh1_el1_df64_global_temp/checkpoint_{k}.pth"))
            else:
                model.load_state_dict(torch.load(f"./checkpoints/v1_iTransformer_Meteorology_ftMS_sl168_ll1_pl24_dm64_nh1_el1_df64_global_wind/checkpoint_{k}.pth"))
            outputs = model(data)
            outputs = outputs[:, :, -1:].detach().cpu().numpy() # (N * S, P, 1)
            P = outputs.shape[1]
            forecast = outputs.reshape(N, S, P, 1) # (N, S, P, 1)
            forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 1)
            if i == 0:
                np.save(os.path.join(save_path, f"temp_predict_{k}.npy"), forecast)
            else:
                np.save(os.path.join(save_path, f"wind_predict_{k}.npy"), forecast)
    w1 = 0.1
    w2 = 0.15
    w3 = 0.35
    w4 = 0.4
    data_temp_1 = np.load(os.path.join(save_path, "temp_predict_0.npy"))
    data_temp_2 = np.load(os.path.join(save_path, "temp_predict_1.npy"))
    data_temp_3 = np.load(os.path.join(save_path, "temp_predict_2.npy"))
    data_temp_4 = np.load(os.path.join(save_path, "temp_predict_3.npy"))
    data_temp = w1*data_temp_1+w2*data_temp_2+w3*data_temp_3+w4*data_temp_4
    np.save(os.path.join(save_path, "temp_predict.npy"), data_temp)
    
    
    data_wind_1 = np.load(os.path.join(save_path, "wind_predict_0.npy"))
    data_wind_2 = np.load(os.path.join(save_path, "wind_predict_1.npy"))
    data_wind_3 = np.load(os.path.join(save_path, "wind_predict_2.npy"))
    data_wind_4 = np.load(os.path.join(save_path, "wind_predict_3.npy"))
    data_wind = w1*data_wind_1+w2*data_wind_2+w3*data_wind_3+w4*data_wind_4
    np.save(os.path.join(save_path, "wind_predict.npy"), data_wind)