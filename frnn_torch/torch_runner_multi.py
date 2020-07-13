from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchsummary import summary

import numpy as np
import sys
if sys.version_info[0] < 3:
    from itertools import imap

#leading to import errors:
#from hyperopt import hp, STATUS_OK
#from hyperas.distributions import conditional

import time
import datetime
import os
from functools import partial
import pathos.multiprocessing as mp

from conf import conf
from loader import Loader, ProcessGenerator
from performance import PerformanceAnalyzer
from evaluation import *
from downloading import makedirs_process_safe


import hashlib

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
from torch.nn.utils import weight_norm
from convlstmnet import *

model_filename = 'torch_model.pt'

class FLSTM(nn.Module):
  def __init__(self,input_dim=14,output_dim=1,rnn_size=200,dense_size=8,dropout=0.1,batch_first=True,bidirectional=False,rnn_layers=2,profile_dim=0,device=None): 
      super(FLSTM, self).__init__()
      pre_rnn = [nn.Linear(input_dim, dense_size)]
      self.pre_rnn_network = nn.Sequential(*pre_rnn)
      self.input_dim=input_dim
      self.rnn_size=rnn_size
      self.dropout=dropout
      self.rnn_layers=rnn_layers
      self.output_dim=output_dim
      if device==None:
         self.device=torch.device('cuda')
      else: 
         self.device=device
      self.rnn=nn.LSTM(self.input_dim,self.rnn_size,batch_first=True,num_layers=self.rnn_layers).to(self.device)
      self.dropout_layer=nn.Dropout(p=self.dropout)
      self.final_linear=nn.Linear(self.rnn_size,self.output_dim).to(self.device)
  def forward(self, x):
     #   x = self.pre_rnn_network(x)
     #   x,_ = nn.LSTM(self.input_dim,self.rnn_size,batch_first=True,dropout=self.dropout,num_layers=self.rnn_layers).to(self.device)(x)
        y, _ = self.rnn(x)
        x = y
        x = self.dropout_layer(x)
        x = self.final_linear(x)
        return x



class FTCN(nn.Module):
    def __init__(self,n_scalars,n_profiles,profile_size,layer_sizes_spatial,
                 kernel_size_spatial,linear_size,output_size,
                 num_channels_tcn,kernel_size_temporal,dropout=0.1):
        super(FTCN, self).__init__()
        self.lin = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout)
        self.input_layer = TimeDistributed(self.lin,batch_first=True)
        self.tcn = TCN(linear_size, output_size, num_channels_tcn , kernel_size_temporal, dropout)
        self.model = nn.Sequential(self.input_layer,self.tcn)
    
    def forward(self,x):
        return self.model(x)


class InputBlock(nn.Module):
    def __init__(self, n_scalars, n_profiles,profile_size, layer_sizes, kernel_size, linear_size, dropout=0.2):
        super(InputBlock, self).__init__()
        self.pooling_size = 2
        self.n_scalars = n_scalars
        self.n_profiles = n_profiles
        self.profile_size = profile_size
        self.conv_output_size = profile_size
        if self.n_profiles == 0:
            self.net = None
            self.conv_output_size = 0
        else:
            self.layers = []
            for (i,layer_size) in enumerate(layer_sizes):
                if i == 0:
                    input_size = n_profiles
                else:
                    input_size = layer_sizes[i-1]
                self.layers.append(weight_norm(nn.Conv1d(input_size, layer_size, kernel_size)))
                self.layers.append(nn.ReLU())
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,self.pooling_size,self.pooling_size)
                self.layers.append(nn.Dropout2d(dropout))
            self.net = nn.Sequential(*self.layers)
            self.conv_output_size = self.conv_output_size*layer_sizes[-1]
        self.linear_layers = []
        
        print("Final feature size = {}".format(self.n_scalars + self.conv_output_size))
        self.linear_layers.append(nn.Linear(self.conv_output_size+self.n_scalars,linear_size))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Linear(linear_size,linear_size))
        self.linear_layers.append(nn.ReLU())
        print("Final output size = {}".format(linear_size))
        self.linear_net = nn.Sequential(*self.linear_layers)

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if self.n_profiles == 0:
            full_features = x#x_scalars
        else:
            if self.n_scalars == 0:
                x_profiles = x
            else:
                x_scalars = x[:,:self.n_scalars]
                x_profiles = x[:,self.n_scalars:]
            x_profiles = x_profiles.contiguous().view(x.size(0),self.n_profiles,self.profile_size)
            profile_features = self.net(x_profiles).view(x.size(0),-1)
            if self.n_scalars == 0:
                full_features = profile_features
            else:
                full_features = torch.cat([x_scalars,profile_features],dim=1)
                
        out = self.linear_net(full_features)
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
        return out


def calculate_conv_output_size(L_in,padding,dilation,stride,kernel_size):
    return int(np.floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)*1.0/stride + 1))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#dimensions are batch,channels,length
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
#         self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)#.transpose(1,2)).transpose(1,2)
        return output
#         return self.sig(output)




# def train(model,data_gen,lr=0.001,iters = 100):
#     log_step = int(round(iters*0.1))
#     optimizer = opt.Adam(model.parameters(),lr = lr)
#     model.train()
#     total_loss = 0
#     count = 0
#     loss_fn = nn.MSELoss(size_average=False)
#     for i in range(iters):
#         x_,y_,mask_ = data_gen() 
# #         print(y)
#         x, y, mask = Variable(torch.from_numpy(x_).float()), Variable(torch.from_numpy(y_).float()),Variable(torch.from_numpy(mask_).byte())
# #         print(y)
#         optimizer.zero_grad()
# #         output = model(x.unsqueeze(0)).squeeze(0)
#         output = model(x)#.unsqueeze(0)).squeeze(0)
#         output_masked = torch.masked_select(output,mask)
#         y_masked = torch.masked_select(y,mask)
# #         print(y.shape,output.shape)
#         loss = loss_fn(output_masked,y_masked)
#         total_loss += loss.data[0]
#         count += output.size(0)

# #         if args.clip > 0:
# #             torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
#         loss.backward()
#         optimizer.step()
#         if i > 0 and i % log_step == 0:
#             cur_loss = total_loss / count
#             print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(0,lr, cur_loss))
#             total_loss = 0.0
#             count = 0








class TimeDistributed(nn.Module):
    def __init__(self, module,is_half=False, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.is_half=is_half
    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)
        x=x.float()

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        if self.is_half:
         y=y.half()
        return y





def build_torch_model(conf):
    dropout = conf['model']['dropout_prob']
# dim = 10

    # lin = nn.Linear(input_size,intermediate_dim)
    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    print('n_scalars,n_profiles,profile_size=',n_scalars,n_profiles,profile_size)
    dim = n_scalars+n_profiles*profile_size
    input_size = dim
    output_size = 1
    # intermediate_dim = 15
    try:
      layer_sizes_spatial = conf['model']['layer_size_spatial']#[40,20,20]
      kernel_size_spatial = conf['model']['kernel_size_spatial']
    except:
      layer_sizes_spatial = 16
      kernel_size_spatial= 3
    linear_size = 5
    try:
      num_channels_tcn = [conf['model']['tcn_hidden']]*conf['model']['tcn_layers']#[3]*5
      kernel_size_temporal = conf['model']['kernel_size_temporal'] #3
    except:
      num_channels_tcn =  [40]*5 # [conf['model']['tcn_hidden']]*conf['model']['tcn_layers']#[3]*5
      kernel_size_temporal = 3 #conf['model']['kernel_size_temporal'] #3
    try:
      model_type = conf['model']['model_type']
    except:
      model_type='LSTM'
    
    if model_type == 'TCN':  
        model = FTCN(n_scalars,n_profiles,profile_size,layer_sizes_spatial,
             kernel_size_spatial,linear_size,output_size,num_channels_tcn,
             kernel_size_temporal,dropout)
    elif model_type == 'LSTM':
         rnn_size=conf['model']['rnn_size']
         rnn_layers=conf['model']['rnn_layers']
         model=FLSTM(input_dim=dim,output_dim=1,rnn_layers=rnn_layers,rnn_size=rnn_size,dropout=dropout,batch_first=True,bidirectional=False)
    
    elif model_type == 'TTLSTM':
        try:
          tt_dense = conf['model']['tt_lstm_hidden']
          cell_order = conf['model']['cell_order']
          cell_steps = conf['model']['cell_steps']
          cell_rank = conf['model']['cell_rank']
        except:
           tt_dense=20
           cell_order=2
           cell_steps=2
           cell_rank=2
        model = ConvLSTMNet(
        input_channels = 1, 
        layers_per_block = (1,1), 
        hidden_channels = (tt_dense, tt_dense), 
        skip_stride = None,
        cell = 'convttlstm', cell_params = {"order": cell_order,
        "steps": cell_steps, "rank": cell_rank },
        kernel_size = 1, bias = True,
        output_sigmoid = True)

    else:
        print('!!!!!!!!!!!!Architecture NOT implemented.')
        exit(1)
 
    return model





def get_signal_dimensions(conf):
    #make sure all 1D indices are contiguous in the end!
    use_signals = conf['paths']['use_signals']
    n_scalars = 0
    n_profiles = 0
    profile_size = 0
    is_1D_region = use_signals[0].num_channels > 1#do we have any 1D indices?
    for sig in use_signals:
        num_channels = sig.num_channels
        if num_channels > 1:
            profile_size = num_channels
            n_profiles += 1
            is_1D_region = True
        else:
            assert(not is_1D_region), "make sure all use_signals are ordered such that 1D signals come last!"
            assert(num_channels == 1)
            n_scalars += 1
            is_1D_region = False
    return n_scalars,n_profiles,profile_size 

def apply_model_to_np(model,x,device=None):
    #     return model(Variable(torch.from_numpy(x).float()).unsqueeze(0)).squeeze(0).data.numpy()
    return model(Variable(torch.from_numpy(x).float()).to(device)).to(torch.device('cpu')).data.numpy()



def make_predictions(conf,shot_list,loader,custom_path=None,inference_model=None,device=None):
    generator = loader.inference_batch_generator_full_shot(shot_list)
    if inference_model == None:
      if custom_path == None:
        model_path = get_model_path(conf)
      else:
        model_path = custom_path
      print('model-path is: ',model_path)
      inference_model = build_torch_model(conf)
      inference_model.load_state_dict(torch.load(model_path))
      inference_model.to(device)
    #shot_list = shot_list.random_sublist(10)
    inference_model.eval()
    y_prime = []
    y_gold = []
    disruptive = []
    num_shots = len(shot_list)

    while True:
        x,y,mask,disr,lengths,num_so_far,num_total = next(generator)
        #x, y, mask = Variable(torch.from_numpy(x_).float()), Variable(torch.from_numpy(y_).float()),Variable(torch.from_numpy(mask_).byte())
        output = apply_model_to_np(inference_model,x,device=device)
        for batch_idx in range(x.shape[0]):
            curr_length = lengths[batch_idx]
            y_prime += [output[batch_idx,:curr_length,0]]
            y_gold += [y[batch_idx,:curr_length,0]]
            disruptive += [disr[batch_idx]]
        if len(disruptive) >= num_shots:
            y_prime = y_prime[:num_shots]
            y_gold = y_gold[:num_shots]
            disruptive = disruptive[:num_shots]
            break
    return y_prime,y_gold,disruptive

def make_predictions_and_evaluate_gpu(conf,shot_list,loader,custom_path = None,inference_model=None,device=None):
    y_prime,y_gold,disruptive = make_predictions(conf,shot_list,loader,custom_path,inference_model=inference_model,device=device)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss


def get_model_path(conf):
    return conf['paths']['model_save_path']  + model_filename #save_prepath + model_filename


def train_epoch(model,data_gen,optimizer,loss_fn,device=None):
    loss = 0
    total_loss = 0
    num_so_far = 0
    x_,y_,mask_,num_so_far_start,num_total = next(data_gen)
    num_so_far = num_so_far_start
    step = 0
    while True:
        x, y, mask = Variable(torch.from_numpy(x_).float()).to(device), Variable(torch.from_numpy(y_).float()).to(device),Variable(torch.from_numpy(mask_).byte()).to(device)
        optimizer.zero_grad()
        output = model(x)
        output_masked = torch.masked_select(output,mask)
        y_masked = torch.masked_select(y,mask)
        print('INPUTSHAPING::')
        print('x.shape,',x.shape)
        print('OUTPUTSHAPING::')
        print('y.shape:',y.shape)
        print('output.shape:',output.shape)
        loss = loss_fn(output_masked,y_masked)
        total_loss += loss.data.item()
        
        loss.backward()
        optimizer.step()
        step += 1
        print("[{}]  [{}/{}] loss: {:.3f}, ave_loss: {:.3f}".format(step,num_so_far-num_so_far_start,num_total,loss.data.item(),total_loss/step))
        if num_so_far-num_so_far_start >= num_total:
            break
        x_,y_,mask_,num_so_far,num_total = next(data_gen)
    return step,loss.data.item(),total_loss,num_so_far,1.0*num_so_far/num_total


def train(conf,shot_list_train,shot_list_validate,loader):

    np.random.seed(1)
    use_cuda=True #False
    device = torch.device("cuda")

    #data_gen = ProcessGenerator(partial(loader.training_batch_generator_full_shot_partial_reset,shot_list=shot_list_train)())
    data_gen = partial(loader.training_batch_generator_full_shot_partial_reset,shot_list=shot_list_train)()


    loader.set_inference_mode(False)

    train_model = build_torch_model(conf)
    print(train_model)
    train_model.to(device)
   # try:
      #summary(train_model,(500,14))
   # except:
   #   print('MODEL SUMMARY WARNING!!!!!!!!!!!!!!!!!NOT PASSED for some reason.....')

    num_epochs = conf['training']['num_epochs']
    patience = conf['callbacks']['patience']
    lr_decay = conf['model']['lr_decay']
    lr_decay_factor = conf['model']['lr_decay_factor']
    lr_decay_patience = conf['model']['lr_decay_patience']
    batch_size = conf['training']['batch_size']
    lr = conf['model']['lr']
    clipnorm = conf['model']['clipnorm']
    e = 0


    
    if conf['callbacks']['mode'] == 'max':
        best_so_far = -np.inf
        cmp_fn = max
    else:
        best_so_far = np.inf
        cmp_fn = min
    optimizer = opt.Adam(train_model.parameters(),lr = lr)
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer,lr_decay)
    train_model.train()
    not_updated = 0
    total_loss = 0
    count = 0
    loss_fn = nn.MSELoss(size_average=True)
    model_path = get_model_path(conf)
    makedirs_process_safe(os.path.dirname(model_path))
    epochlog=open('epoch_train_log.txt','w')
    epochlog.write('e,         Train Loss,          Val Loss,          Val ROC\n')
    epochlog.close()
    while e < num_epochs-1:
        print('{} epochs left to go'.format(num_epochs - 1 - e))
        print('\nTraining Epoch {}/{}'.format(e,num_epochs),'starting at',datetime.datetime.now())
        train_model.train()
        scheduler.step()
        (step,ave_loss,curr_loss,num_so_far,effective_epochs) = train_epoch(train_model,data_gen,optimizer,loss_fn,device=device)
        e = effective_epochs
    
        print('\nFiniehsed Training'.format(e,num_epochs),'finishing at',datetime.datetime.now())
        loader.verbose=False #True during the first iteration
        print('printing_out epoch ', e,'learning rate:',lr)
        for param_group in optimizer.param_groups:
             print(param_group['lr'])

        _,_,_,roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader,inference_model=train_model,device=device)
        best_so_far = cmp_fn(roc_area,best_so_far)

        stop_training = False
        print('=========Summary======== for epoch{}'.format(step))
        print('Training Loss numpy: {:.3e}'.format(ave_loss))
        print('Validation Loss: {:.3e}'.format(loss))
        print('Validation ROC: {:.4f}'.format(roc_area))
        epochlog=open('epoch_train_log.txt','a')
        epochlog.write(str(e)+'  '+str(ave_loss)+'   ' +str(loss)+'  '+str(roc_area) +'\n')
        epochlog.close()
        if best_so_far != roc_area: #only save model weights if quantity we are tracking is improving
            print("No improvement, still saving model")
            not_updated += 1
            
            if e > 10 and not_updated>=lr_decay_patience:
                lr /=lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            print("Saving model")
            not_update=0
            # specific_builder.delete_model_weights(train_model,int(round(e)))
################Saving torch model################################
            torch.save(train_model.state_dict(),model_path)
            torch.save(train_model,model_path[:-3]+'full_model.pt')
##################################################################
        if not_updated > patience:
            print("Stopping training due to early stopping")
            break

