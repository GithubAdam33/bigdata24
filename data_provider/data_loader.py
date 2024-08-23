import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, size=None, features='MS'):
        self.seq_len = size[0] # inference length: 24 * 7 = 168
        self.label_len = size[1] # label: 1(training temp and wind seperately)
        self.pred_len = size[2] # prediction length: 24

        self.features = features
        self.root_path = root_path
        # self.data_path = data_path
        self.data_path_w = 'wind.npy'
        self.data_path_t = 'temp.npy'
        self.data= self.__read_data__()
        # self.stations_num = self.data_x.shape[-1] # 3850
        self.stations_num = self.data.shape[1] # 3850
        self.tot_len = len(self.data) - self.seq_len - self.pred_len + 1 # tot_len = 17544 - 168 - 24 + 1 = 17353(ignore the last data)

    def __read_data__(self):
        data_w = np.load(os.path.join(self.root_path, self.data_path_w)) # (T, S, 1)
        data_t = np.load(os.path.join(self.root_path, self.data_path_t))
        data = np.concatenate((data_w, data_t), axis=2) #(t,s,2)
        
        data_t_diff = np.diff(data_t, axis=0)
        data_w_diff = np.diff(data_w, axis=0)
        pad_width = [(1, 0)] + [(0, 0)]*(data_t.ndim - 1)
        data_t_diff = np.pad(data_t_diff, pad_width, mode='constant', constant_values=0)
        data_w_diff = np.pad(data_w_diff, pad_width, mode='constant', constant_values=0)
        data_tw = np.concatenate((data_t_diff, data_w_diff), axis=2) #(t,s,2)
        data_tw = data_tw.transpose(0, 2, 1) # (T, 2, S)
        
        data = data.transpose(0, 2, 1) # (T, 2, S)
        
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy'))
        ## make extra feature
        sqrt_sum = np.sqrt(np.sum(era5[:, :2, :, :]**2, axis=1, keepdims=True))
        mean_sum = np.mean(era5, axis=2, keepdims=True)
        sqrt_sum = sqrt_sum.reshape(sqrt_sum.shape[0], -1, sqrt_sum.shape[3])
        mean_sum = mean_sum.reshape(mean_sum.shape[0], -1, mean_sum.shape[3])
        new_data = np.concatenate((sqrt_sum, mean_sum), axis=1) # (T, 2, S)
        new_data = np.repeat(new_data, 3, axis=0)[:len(data), :, :] #(T,13,S)
        # interpolate
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S), each covariate repeats 3 times, align
        #
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S) /
        data = np.concatenate((repeat_era5,new_data,data_tw,data), axis=1) # (T, 36+13+2+2, S)  53
        
        
        ##
        return data

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len # the begin and end index of data input for model, 0 ~ 168
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len # the begin and end index of label, 168 - 1 ~ 168 + 24
        
        seq_x = self.data[s_begin:s_end, :, station_id]# (168, 36)
        seq_y = self.data[r_begin:r_end, :, station_id] # (25, 36)
        
        return seq_x, seq_y

    def __len__(self):
        return (len(self.data) - self.seq_len - self.pred_len + 1) * self.stations_num # total len