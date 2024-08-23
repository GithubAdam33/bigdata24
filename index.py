import os
import numpy as np
import random
import torch
from models import iTransformer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
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
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 2,
        'pred_len': 72,
        'enc_in': 38,
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
    
    ##后面改index在这里改参数就行
    Num_epoch = 3
    Num_K_fold = 5
    
    
    for i in range(Num_epoch):
        data_t = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
        data_w = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
        N, L, S, _ = data_t.shape
        data = np.concatenate([data_w, data_t], axis=3)  # 72, 168, 60, 2

        cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy")) 
        
        repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)
        C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
        covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
        data = data.transpose(0, 1, 3, 2) # (N, L, 2, S)
        C = C1 + 2
        data = np.concatenate([covariate, data], axis=2) # (N, L, C=38, S)
        data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
        data = data.reshape(N * S, L, C)
        data = torch.tensor(data).float().cuda() # (N * S, L, C)

        # k-fold
        k_folds = Num_K_fold
        
        
        for fold in range(k_folds):
            model = iTransformer.Model(args).cuda()
            state_dict = torch.load(f"./checkpoints/v1_iTransformer_Meteorology_ftM_sl168_ll2_pl72_dm64_nh1_el1_df64_global_temp_wind_kfold/checkpoint_fold{fold}_epoch{i}.pth")
            new_state_dict = {}
            for key in list(state_dict.keys()):
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            model.load_state_dict(new_state_dict)
            model.eval()

            with torch.no_grad():
                output_w,output_t = model(data)
                output_w = output_w[:, :, -2:-1].detach().cpu().numpy() # (N * S, P, 1)
                output_t = output_t[:, :, -1:].detach().cpu().numpy() # (N * S, P, 1)
                P = output_w.shape[1]
                forecast_w = output_w.reshape(N, S, P, 1) # (N, S, P, 1)
                forecast_w = forecast_w.transpose(0, 2, 1, 3) # (N, P, S, 1)
                forecast_t = output_t.reshape(N, S, P, 1) # (N, S, P, 1)
                forecast_t = forecast_t.transpose(0, 2, 1, 3) # (N, P, S, 1)
            
            np.save(os.path.join(save_path, f"wind_predict_fold{fold}_epoch{i}.npy"), forecast_w)
            np.save(os.path.join(save_path, f"temp_predict_fold{fold}_epoch{i}.npy"), forecast_t)
    
    
    #epoch权重  
    weight = [0, 1, 0, 0.35, 0.2]
    
    data_temp_l = []
    data_wind_l = []
    for i in range(Num_K_fold):
        data_t = np.load(os.path.join(save_path, f"temp_predict_fold{i}_epoch0.npy"))*weight[0]
        data_w = np.load(os.path.join(save_path, f"wind_predict_fold{i}_epoch0.npy"))*weight[0]
        for k in range(1,Num_epoch):
            data_t += np.load(os.path.join(save_path, f"temp_predict_fold{i}_epoch{k}.npy"))*weight[k]
            data_w += np.load(os.path.join(save_path, f"wind_predict_fold{i}_epoch{k}.npy"))*weight[k]
        data_temp_l.append(data_t)
        data_wind_l.append(data_w)

    data_temp = np.mean(data_temp_l, axis=0)
    data_wind = np.mean(data_wind_l, axis=0)
    
    np.save(os.path.join(save_path, "temp_predict.npy"), data_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), data_wind)