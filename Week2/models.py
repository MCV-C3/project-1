
import torch.nn as nn
import torch

from typing import *

class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, hidden_layers_n:int ,output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d
        self.hidden_layers_n = hidden_layers_n
        self.activation = nn.ReLU() 

        layers = []

        layers.append(nn.Linear(input_d, hidden_d))

        for i in range(hidden_layers_n):
            layers.append(nn.Linear(hidden_d, hidden_d))

        layers.append(nn.Linear(hidden_d, output_d))

        self.layers = nn.ModuleList(layers)
        

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        for layer in  self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        
        x = self.layers[-1](x)
        
        return x


    def recover_layer(self,x,layer_int):

        if layer_int == self.hidden_layers_n+1:
            return forward(x)

        else:
            x = x.view(x.shape[0], -1)

        for layer in  self.layers[:layer_int]:
            x = layer(x)
            x = self.activation(x)
        return x
