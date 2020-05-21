import pytorch
from pytorch import nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

global device 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#reproductbility
np.random.seed(30191375)
torch.manual_seed(30191375)
torch.cuda.manual_seed(30191375)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn',force = True)

class Layer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    def forward(self, FROM):
        TO = self.linear(FROM)
        TO = self.activation(TO)
        return TO

class AutoEncoder(nn.Module):
    def __init__(self,in_dim,layers=[2500],activation=F.tanh,loss_func = nn.MSELoss(reduce=True,size_average=True),optmizer = optim.Adam):
        super(AutoEncoder,self).__init__()
        self.activation = activation
        self.layers = [in_dim,*layers]
        self.l = len(self.layers)-1
        self.loss_func = loss_func
        self.optmizer = optmizer
        self.encoder = nn.Sequential(
            [Layer(self.layers[i],self.layers[i+1],self.activation) for i in range(self.l)]
        )
        self.decoder = nn.Sequential(
            [Layer(self.layers.reverse()[i],self.layers.reverse()[i+1],self.activation) for i in range(self.l)]
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

    


# pretraining
loss_func = nn.MSELoss(reduce=True,size_average=True)

# fine-tuning