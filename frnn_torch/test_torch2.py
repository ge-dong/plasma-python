import torch
import torch.nn as nn
from torchsummary import summary

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


model=model1()
model.to(device)
print(model)
inputs = torch.randn(1,200, 20).to(device)
print(inputs.size())

outputs=model(inputs)
print('Finished output')


print(outputs.size())

#summary(model,(500,20))





