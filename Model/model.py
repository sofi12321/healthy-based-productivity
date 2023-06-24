import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class I_Filter:
    #TODO
    pass

class O_Filter:
    def __init__(self, out_features):
        self.out_features = out_features

    def check_overlay(self, time, free_time_slots):
        #TODO
        return True


class SC_LSTM(nn.Module):

    def __init__(self, in_features, out_features, layers, batch_size=1, device='cpu', dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(SC_LSTM, self).__init__()
        # Variables declaration
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.batch_size = batch_size
        self.hn = torch.zeros(layers, batch_size, out_features)         # LSTM intermediate result
        self.cn = torch.zeros(layers, batch_size, out_features)          # LSTM intermediate result

        # Objects declaration
        self.i_f = I_Filter()
        self.o_f = O_Filter(out_features)
        self.lstm = nn.LSTM(in_features, out_features, layers, batch_first=True)
        self.activation = nn.Linear(out_features, layers)


    def forward(self, x, free_time_slots):
        #TODO: check x for input filtration


        # Check for the output filtration until it give feasible solution
        while True:
            out, (hn_new, cn_new) = self.lstm(x, (self.hn, self.cn))
            out_check = self.o_f.check_overlay(out, free_time_slots)

            if out_check:
                self.hn = hn_new
                self.cn = cn_new
                return out

            else:
                if not self.training:
                    #TODO: remove or deatach the last block during inference
                    pass




