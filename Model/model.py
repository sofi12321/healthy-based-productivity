import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class I_Filter:
    def __init__(self, model):
        """
        Simple initialization of the class
        :param model: the initial model object
        """
        self.model = model

    def filter_task(self, x, task_type):
        """
        This function change the model behavior depending on the task type.
        :param x: input feature vector
        :param task_type: 'resched', 'non-resched' need fo varying forward pass
        :return: y: output feature vector in case of non-reschedulable task or
        y and model states in case of reschedulable task (for further output filtration).
        """
        # If the task is not reschedulable,
        if task_type == 'non-resched':
            h_new = self.model.injector(x)

            # Forward the model but making it "thinking" that it returns h_new
            _, (_, cn_new) = self.model.lstm(x, (self.model.hn, self.model.cn))
            self.model.hn = h_new
            self.model.cn = cn_new

            # Return not modified task times
            #TODO: parse task into start, end and refractory times.
            y = (start, end, refr)
            return y

        # If the task is reschedulable, just forward the lstm and linear layers
        elif task_type == 'resched':
            out, (hn_new, cn_new) = self.model.lstm(x, (self.model.hn, self.model.cn))
            out = self.model.lstm_linear(out)
            return out, (hn_new, cn_new)





class O_Filter:
    def __init__(self, out_features, model):
        """

        :param out_features:
        :param model:
        """
        self.out_features = out_features
        self.model = model


    def filter_task(self, x, out, model_state, free_time_slots):
        """
        This function change the model behavior depending on its output.
        :param x: input feature vector
        :param out: initial model prediction from input filtration.
        :param free_time_slots: free time available in next `pred_interval` to insert the task.
        :param model_state: model states (hidden and cell states)
        :return: y: output feature vector
        """
        (hn_new, cn_new) = model_state

        # Check for the output filtration until it give feasible solution
        while True:
            out_check = self._check_overlay(out, free_time_slots)

            # If the output is feasible, set new hidden and cell states and return the output
            if out_check:
                self.model.hn = hn_new
                self.model.cn = cn_new
                return out

            # Else forward the model with the previous hidden and cell states
            #TODO: maybe slightly modify h or c or x for network to not create the same output
            out = self.model.lstm(x, (self.model.hn, self.model.cn))
            out = self.model.lstm_linear(out)


    def _check_overlay(self, times, free_time_slots):
        start, end, refr = times
        #TODO
        return True





class SC_LSTM(nn.Module):

    def __init__(self, in_features, lstm_layers, hidden, out_features, pred_interval=1440, hidden_injector=40, device='cpu', dtype=None):
        """
        A constructor for the SC_LSTM class.
        :param in_features: number of input features
        :param lstm_layers: number of layers in LSTM
        :param hidden: size of hidden layer LSTM would output
        :param out_features: number of output features after LSTM -> Linear layer
        :param pred_interval: interval where network should schedule the task
        :param hidden_injector: size of hidden layer in injector network.
            Used in non-reschedulable tasks to inject context to LSTM.
        :param device: device to run the model on.
        :param dtype: data type to run the model on.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(SC_LSTM, self).__init__()
        # Variables declaration
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.lstm_layers = lstm_layers
        self.hidden = hidden
        self.pred_interval = pred_interval             # Prediction interval in minutes (24*60=1440)
        self.hn = torch.zeros(lstm_layers, 1, hidden)         # LSTM intermediate result
        self.cn = torch.zeros(lstm_layers, 1, hidden)          # LSTM intermediate result

        # Objects declaration
        self.i_f = I_Filter(self)
        self.o_f = O_Filter(out_features, self)
        self.lstm = nn.LSTM(in_features, hidden, lstm_layers, batch_first=True)
        self.lstm_linear = nn.Linear(hidden, out_features)

        # NN for injecting non reschedulable tasks
        self.hidden_injector = hidden_injector
        self.injector = nn.Sequential(
            nn.Linear(in_features, hidden_injector),
            nn.ReLU(),
            nn.Linear(hidden_injector, hidden),
            nn.ReLU()
        )



    def forward(self, x, task_type, free_time_slots):
        """
        Forward pass of the model.
        :param x: input feature vector.
        :param task_type: 'resched', 'non-resched'.
        :param free_time_slots: free time available in next `pred_interval` to insert the task.
        :return: y: output feature vector.
        """

        # Check for the input filtration if model is in evaluation mode
        if not self.training:
            out, (hn_new, cn_new) = self.i_f.filter_task(x, task_type)

            if task_type == 'non-resched':
                return out

            elif task_type == 'resched':
                out = self.o_f.filter_task(x, out, (hn_new, cn_new), free_time_slots)
                return out





