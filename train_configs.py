import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import yaml

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Demo')

    # basic config
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # supplementary
    parser.add_argument('--params', type=str, default='./params/config_1.yaml', help='config file path')
    parser.add_argument('--root_path_val', type=str, default='/root/mine/BDC2024/dataset/north/val/', help='root path of the evaluation data file') 
    parser.add_argument('--data_path_val', type=str, default='temp.npy', help='evaluation data file')
    parser.add_argument('--steps_val', type=int, default=1000, help='evaluation steps')
     
    args = parser.parse_args()
    
    # load and cover the default config
    with open(args.params, 'r') as file:
        config = yaml.safe_load(file)

    args.root_path = config['root_path']
    args.data_path = config['data_path']
    args.model_id = config['model_id']
    args.model = config['model']
    args.data = config['data']
    args.features = config['features']
    args.seq_len = config['seq_len']
    args.label_len = config['label_len']
    args.pred_len = config['pred_len']
    args.e_layers = config['e_layers']
    args.enc_in = config['enc_in']
    args.d_model = config['d_model']
    args.d_ff = config['d_ff']
    args.n_heads = config['n_heads']
    args.des = config['des']
    args.learning_rate = config['learning_rate']
    args.batch_size = config['batch_size']
    args.train_epochs = config['train_epochs']
    args.num_workers = config['num_workers']
    args.root_path_val = config['root_path_val']
    args.data_path_val = config['data_path_val']
    args.steps_val = config['steps_val']
    args.lradj = config["lradj"]
     
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast
    
    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.des, ii)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)