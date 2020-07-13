import torch
import torch.nn as nn
from torchsummary import summary

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






device = torch.device("cuda")
class model1(nn.Module):
  def __init__(self): 
      super(model1, self).__init__()
      m1 = nn.Linear(20, 14)
      m2 = nn.Linear(14, 14)
     # m2 = nn.LSTM(14,128,batch_first=True)
  
      self.network = nn.Sequential(m1,m2)#.to(device)
      self.lstm=nn.LSTM(14,128,batch_first=True).to(device)
      self.final_linear=nn.Linear(128,1).to(device) 
  def forward(self, x):
        x=self.network(x)
        x,_=self.lstm(x)
        x=self.final_linear(x)
        return x


class FLSTM(nn.Module):
  def __init__(self,input_dim=14,output_dim=1,rnn_size=128,dense_size=16,dropout=0.01,batch_first=True,bidirectional=False,profile_dim=0,device=None):
      super(FLSTM, self).__init__()
      pre_rnn = [nn.Linear(input_dim, dense_size)]
      self.pre_rnn_network = nn.Sequential(*pre_rnn)
      self.input_dim=input_dim
      self.rnn_size=rnn_size
      self.output_dim=output_dim
      if device==None:
         self.device=torch.device('cuda')
      else:
         self.device=device
      self.rnn=nn.LSTM(self.input_dim,self.rnn_size,batch_first=True).to(self.device)
      self.final_linear=nn.Linear(self.rnn_size,self.output_dim).to(self.device)
  def forward(self, x):
     #   x = self.pre_rnn_network(x)
        x,_ = self.rnn(x)
        x = self.final_linear(x)
        return x





#model=model1()
model=FLSTM(input_dim=14,output_dim=1,dropout=0.1,batch_first=True,bidirectional=False)
model.to(device)
print(model)
inputs = torch.randn(2,500, 14).to(device)
print(inputs.size())
try:
  summary(model,(500,14))
except:
  pass
outputs=model(inputs)
print('Finished output')


print(outputs.size())






