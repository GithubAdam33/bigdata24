from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import os
import time
import warnings
import numpy as np
import wandb

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # Initialize WandB
        wandb.init(project="BDC2024-iTransformer", entity="ljcheng136", config=args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        data_set, data_loader = data_provider(self.args)
        return data_set, data_loader

    def _get_data_val(self):
        args_val = self.args
        args_val.root_path = args_val.root_path_val
        args_val.data_path = args_val.data_path_val
        data_set, data_loader = data_provider(args_val)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = []
        criterion = self._select_criterion()
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs_w,outputs_t = self.model(batch_x)
                # f_dim = -1 if self.args.features == 'MS' else 0
                # f_dim = -2
                
                outputs_w = outputs_w[:, -self.args.pred_len:, -2:-1]
                outputs_t = outputs_t[:, -self.args.pred_len:, -1:]
                outputs = torch.cat((outputs_w, outputs_t), axis=2)
                batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        # train_data, train_loader = self._get_data()
        # val_data, val_loader = self._get_data_val()
        train_data, _ = self._get_data()
         
        # K-fold cross validation
        k_folds=5
        kfold = KFold(n_splits=k_folds, shuffle=True)
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
         
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            # initialize model every fold
            self.model = self._build_model().to(self.device)
            
            train_subsampler = Subset(train_data, train_ids)
            val_subsampler = Subset(train_data, val_ids)
            
            num_works = self.args.num_workers
            train_loader = DataLoader(train_subsampler, batch_size=self.args.batch_size, shuffle=True, num_workers=num_works, drop_last=True)
            val_loader = DataLoader(val_subsampler, batch_size=self.args.batch_size, shuffle=False, num_workers=num_works, drop_last=True)
        
            # previous
            time_now = time.time()
            train_steps = len(train_loader)
            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                # train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs_w,outputs_t = self.model(batch_x)[0]
                            else:
                                outputs_w,outputs_t = self.model(batch_x)

                            # f_dim = -1 if self.args.features == 'MS' else 0
                            
                            
                            outputs_w = outputs_w[:, -self.args.pred_len:, -2:-1]
                            outputs_t = outputs_t[:, -self.args.pred_len:, -1:]
                            outputs = torch.cat((outputs_w, outputs_t), axis=2)
                            batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            # train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs_w,outputs_t = self.model(batch_x)[0]
                        else:
                            outputs_w,outputs_t = self.model(batch_x)

                        outputs_w = outputs_w[:, -self.args.pred_len:, -2:-1]
                        outputs_t = outputs_t[:, -self.args.pred_len:, -1:]
                        outputs = torch.cat((outputs_w, outputs_t), axis=2)
                        
                        batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        # train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    # Clip the gradient
                    max_norm = 2
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                        model_optim.step()

                    # Log the training loss to wandb
                    # wandb.log({'train_loss': loss.item()})
                    wandb.log({f'train_loss_{fold}_fold': loss.item()})
                    
                    # Evaluation
                    # if (i + 1) % self.args.steps_val == 0:
                    #     val_loss = self._evaluate(val_loader)
                    #     print("\tEvaluation | Step: {0}, Epoch: {1} | Val Loss: {2:.7f}".format(i + 1, epoch + 1, val_loss))
                    #     wandb.log({'val_loss': val_loss})

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                # train_loss = np.average(train_loss)

                # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                #     epoch + 1, train_steps, train_loss))
                # torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}.pth')
                torch.save(self.model.state_dict(), path + '/' + f'checkpoint_fold{fold}_epoch{epoch}.pth')
                adjust_learning_rate(model_optim, epoch + 1, self.args)
        
            val_loss = self._evaluate(val_loader)
            print(f"Fold {fold}, Val Loss: {val_loss:.7f}")
            wandb.log({'val_loss': val_loss})
        
        wandb.finish()
        return self.model
