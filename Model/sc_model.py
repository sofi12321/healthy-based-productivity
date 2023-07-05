import torch
import torch.nn as nn
from Data.converter import Converter as Conv
import numpy as np


class InFilter:
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
            duration = x[1]
            date = x[3]
            offset = 0
            model_out = self.model.conv.user_to_model(date, duration, offset)

            return model_out

        # If the task is reschedulable, just forward the lstm and linear layers
        elif task_type == 'resched':
            out, (hn_new, cn_new) = self.model.lstm(x, (self.model.hn, self.model.cn))
            out = self.model.lstm_linear(out)
            out = self.model.lstm_lin_activation(out)
            return out, (hn_new, cn_new)





class Out_Filter:
    def __init__(self, out_features, model):
        """
        Simple constructor of the output filter.
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
            out_check = self.__check_overlay(out, free_time_slots)

            # If the output is feasible, set new hidden and cell states and return the output
            if out_check:
                self.model.hn = hn_new
                self.model.cn = cn_new
                return out

            # Else forward the model with the previous hidden and cell states
            #TODO: maybe slightly modify h or c or x for network to not create the same output
            out = self.model.lstm(x, (self.model.hn, self.model.cn))
            out = self.model.lstm_linear(out)


    def __check_overlay(self, times, free_time_slots):
        """
        This function checks if the scheduled task fit in available time slots.
        :param times: tuple of (start, end, refr) times of the scheduled task
        :param free_time_slots: normalized array of
            free time slots [[0.0, 0.02], [0.07, 0.2], ...]
        :return:
        """
        pred_interval = self.model.pred_interval
        start, end, refr = times

        # Check if the scheduled task fit in available time slots
        for time_slot in free_time_slots:
            if time_slot[0] <= start <= time_slot[1] and time_slot[0] <= end <= time_slot[1] and start > refr:
                return True

        return False





class SC_LSTM(nn.Module):
    def __init__(self, in_features, lstm_layers, hidden, out_features, batch_size, pred_interval=1440, hidden_injector=40, device='cpu', dtype=None):
        """
        A constructor for the SC_LSTM class.
        :param in_features: number of input features
        :param lstm_layers: number of layers in LSTM
        :param hidden: size of hidden layer LSTM would output
        :param out_features: number of output features after LSTM -> Linear layer
        :param batch_size: size of a batch
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
        self.batch_size = batch_size
        self.train_mode = 'lstm'
        self.pred_interval = pred_interval             # Prediction interval in minutes (24*60=1440)
        self.hn = None
        self.cn = None
        # Reset states of the model to zero while initialize
        self.reset_states()

        # Randomly initialize weights for all the layers
        self.__init_weights()

        # Objects declaration
        self.i_f = InFilter(self)
        self.o_f = Out_Filter(out_features, self)
        self.conv = Conv(self.pred_interval)
        self.lstm = nn.LSTM(in_features, hidden, lstm_layers, batch_first=True)
        self.lstm_linear = nn.Linear(hidden, out_features)
        self.lstm_lin_activation = nn.ReLU()

        # NN for injecting non reschedulable tasks
        self.hidden_injector = hidden_injector
        self.injector = nn.Sequential(
            nn.Linear(in_features, hidden_injector),
            nn.ReLU(),
            nn.Linear(hidden_injector, hidden),
            nn.ReLU()
        )

    def forward(self, x, task_type='', free_time_slots='', save_states=False):
        """
        Forward pass of the model.
        :param x: input feature vector.
        :param task_type: 'resched', 'non-resched'.
        :param free_time_slots: free time available in next `pred_interval` to insert the task.
        :param save_states: if True, save the hidden and cell states of the model.
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

        elif self.training:
            # If we need to train only lstm use lstm and linear layer
            if self.train_mode == "lstm":
                out, (hn, cn) = self.lstm(x, (self.hn, self.cn))

                # Reinitialize hidden and cell states as new hn and cn by copying them
                if save_states:
                    self.hn = hn.detach()
                    self.cn = cn.detach()

                out = self.lstm_linear(out)
                out = self.lstm_lin_activation(out)
                return out

            # If we need to train only injector use injector and freezed pretrained linear
            elif self.train_mode == "injector":
                out = self.injector(x)
                out = self.lstm_linear(out)
                out = self.lstm_lin_activation(out)
                return out



    def train_lstm(self):
        """
        This function sets the model to train only lstm
         and lstm_linear layers. It is first step of training
         where only reschedulable tasks are used.
        :return: None
        """
        # Set injector to not trainable
        for param in self.injector.parameters():
            param.requires_grad = False

        # Unfreeze lstm and lstm_linear layers
        frozen_layers = [self.lstm, self.lstm_linear, self.lstm_lin_activation]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = True

        self.train_mode = "lstm"



    def train_injector(self):
        """
        This function sets the model to train only injector
         and freeze other layers. It is second step of training
         where only non-reschedulable tasks are used.
        :return: None
        """
        # To train injector we need to freeze lstm and lstm_linear layers
        frozen_layers = [self.lstm, self.lstm_linear, self.lstm_lin_activation]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Leave only injector trainable
        for param in self.injector.parameters():
            param.requires_grad = True

        self.train_mode = "injector"



    def eval(self):
        """
        This function overload eval() function of nn.Module
        :return: None
        """
        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False
        # Call the original eval() function
        super(SC_LSTM, self).eval()



    def train(self, mode=True):
        """
        This function overload train() function of nn.Module
        :param mode: хз зачем оно тут, вроде как легоси торча :)
        :return: None
        """
        # Set requires_grad to True for all parameters
        for param in self.parameters():
            param.requires_grad = True

        # Call the original train() function
        super(SC_LSTM, self).train(True)



    def reset_states(self):
        """
        This function resets the hidden and cell states of the model to zeros.
        :return: None
        """
        self.hn = torch.zeros(self.lstm_layers, self.batch_size, self.hidden)         # LSTM intermediate result
        self.cn = torch.zeros(self.lstm_layers, self.batch_size, self.hidden)          # LSTM intermediate result
        if self.batch_size == 1:
            self.hn = self.hn.squeeze(1)
            self.cn = self.cn.squeeze(1)



    def __init_weights(self):
        """
        This function randomly initializes the weights of the model.
        :return: None
        """
        for name, param in self.named_parameters():
            nn.init.normal_(param)


    def get_states(self):
        """
        Returns current hidden and cell states of the model.
        :return:
        """
        return self.hn, self.cn


    def set_states(self, hn, cn):
        """
        Sets hidden and cell states of the model.
        :param hn: hidden state
        :param cn: cell state
        :return: None
        """
        # Convert to tensors with appropriate dtype if they are not
        if not isinstance(hn, torch.Tensor):
            hn = torch.tensor(hn).type(torch.float32)
        if not isinstance(cn, torch.Tensor):
            cn = torch.tensor(cn).type(torch.float32)

        # Set states
        self.hn = hn
        self.cn = cn


