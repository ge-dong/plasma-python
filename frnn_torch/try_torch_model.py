import torch 
from torchsummary import summary
from conf import conf
model_path='./model_checkpoints/torch_modelfull_model.pt'
from torch_runner_dist import build_torch_model
inference_model = build_torch_model(conf)
print(inference_model)
inference_model=torch.load(model_path)
##inference_model.load_state_dict(torch.load(model_path))
#print(inference_model)
use_cuda=True #False
device = torch.device("cuda" if use_cuda else "cpu")

inference_model.to(device)
summary(inference_model,(500,14))
