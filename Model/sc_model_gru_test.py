import torch
import torch.nn as nn
import numpy as np
from Data.converter import Converter as Conv


class SC_GRU(nn.Module):
    def __init__(self, in_features, gru_layers, hidden, out_features, batch_size, pred_interval=1440, hidden_injector=40, device='cpu', dtype=None):
        """
        A constructor for the SC_LSTM class.
        :param in_features: number of input features
        :param gru_layers: number of layers in LSTM
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

        super(SC_GRU, self).__init__()
        # Variables declaration
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.gru_layers = gru_layers
        self.hidden = hidden
        self.batch_size = batch_size
        self.train_mode = 'gru'
        self.pred_interval = pred_interval  # Prediction interval in minutes (24*60=1440)
        self.hn = None
        self.reset_states()     # Reset states of the model to zero while initialize
        self.__init_weights()   # Randomly initialize weights for all the layers

        # Objects declaration
        self.converter = Conv(self.pred_interval)
        self.gru = nn.GRU(in_features, hidden, gru_layers)
        self.gru_linear = nn.Linear(hidden, out_features)
        self.lstm_lin_activation = nn.Sigmoid()

        # NN for injecting non reschedulable tasks
        self.hidden_injector = hidden_injector
        self.injector = nn.Sequential(
            nn.Linear(in_features, hidden_injector),
            nn.ReLU(),
            nn.Linear(hidden_injector, hidden * gru_layers),
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

            if task_type == 'non-resched':
                return self.__plan_non_resched(x, save_states)

            elif task_type == 'resched':
                return self.__plan_resched(x, free_time_slots, save_states)


        elif self.training:
            # If we need to train only lstm use lstm and linear layer
            if self.train_mode == "lstm":
                out, (hn, cn) = self.gru(x, (self.hn, self.cn))

                # Reinitialize hidden and cell states as new hn and cn by copying them
                if save_states:
                    self.hn = hn.detach()
                    self.cn = cn.detach()
                out = self.gru_linear(out)
                out = self.lstm_lin_activation(out)
                return out

            # If we need to train only injector use injector and freezed pretrained linear
            elif self.train_mode == "injector":
                out = self.injector(x)
                out = self.gru_linear(out)
                out = self.lstm_lin_activation(out)
                return out


    def __plan_non_resched(self, x, save_states):
        """
        A function to plan a non-reschedulable task. It didn't check
        the free time slots and simply inject the context to the LSTM and return
        the input as an output.
        :param x: input feature vector
        :param save_states: if True, save the hidden and cell states of the model.
        :return: y: output feature vector.
        """

        if save_states:
            self.__use_injector(x)



    def __plan_resched(self, x, free_time_slots, save_states):
        """
        A function to plan a reschedulable task. It changes
        correspondingly the behavior of the model and filter the output
        to fit the free time slots.
        :param x: input feature vector
        :param free_time_slots: normalized array of
            free time slots [[0.0, 0.02], [0.07, 0.2], ...]
        :param save_states: if True, save the hidden and cell states of the model.
        :return: y: output feature vector.
        """

        out, (hn_new, cn_new) = self.gru(x, (self.hn, self.cn))
        out = self.gru_linear(out)
        out = self.lstm_lin_activation(out)

        # Create a coppy of free time slots
        free_time_slots = np.copy(free_time_slots)
        if self.__check_overlay(out, free_time_slots):
            # TODO: SHOULD THE MODEL USE AN OUTPUT DURATION AND REFR, OR USE USER DEFINED ONES?
            # Unpack torch tensors times into 3 variables and convert them to numpy
            start, duration, refr = out[0][0].item(), out[0][1].item(), out[0][2].item()


            print(f"Intermediary output: {out}")
            total_duration = duration + refr
            mean_predict_pos = np.mean([start, start + total_duration])

            free_time_slots_means = []

            for elem in free_time_slots:
                elem_duration = elem[1] - elem[0]

                # Delete all the slots that are not feasible for the task duration
                if total_duration > elem_duration:
                    free_time_slots = np.delete(free_time_slots, np.where(free_time_slots == elem)[0][0], axis=0)

                # Save mean of the available time slot if it is feasible
                else:
                    free_time_slots_means.append(np.mean(elem))

            # Choose the closest mean to the predicted mean interval (if it is possible)
            try:
                best_time_slot_num = np.argmin(np.abs(np.array(free_time_slots_means) - mean_predict_pos))

                # Get corresponding time slot
                best_time_slot = free_time_slots[best_time_slot_num]

            # If there is no available time slots, return none
            except ValueError:
                return None


            out = (best_time_slot[0], duration, refr)

            if save_states:
                # Convert output (,3) and x (,11) to the input model format (,11)
                new_x = self.converter.out_to_features(x, out)
                self.__use_injector(new_x)
            return out

        # Save the states and return not modified output if it is feasible
        else:
            if save_states:
                self.hn = hn_new.detach()
                self.cn = cn_new.detach()
            return out


    def __use_injector(self, x):
        # Create h_new by an injector
        hn_new = self.injector(x)

        # Reshape hn_new to fit the LSTM input
        hn_new = hn_new.view(self.gru_layers, self.batch_size, self.hidden)
        if self.batch_size == 1:
            hn_new = hn_new.squeeze(1)

        # Forward lstm with to get c_new
        _, (_, cn_new) = self.gru(x, (self.hn, self.cn))

        # Set new injector substituted hidden and new cell state
        self.hn = hn_new.detach()
        self.cn = cn_new.detach()


    def __check_overlay(self, times, free_time_slots):
        """
        This function checks if the scheduled task fit in available time slots.
        :param times: tuple of (start, end, refr) times of the scheduled task
        :param free_time_slots: normalized array of
            free time slots [[0.0, 0.02], [0.07, 0.2], ...]
        :return: True if there is an overlay, False otherwise
        """
        # Unpack torch tensors times into 3 variables
        start = times[0][0].item()
        duration = times[0][1].item()
        refr = times[0][2].item()
        end = start + duration + refr

        # Check if the scheduled task fit in available time slots
        for time_slot in free_time_slots:
            if time_slot[0] <= start <= time_slot[1] and time_slot[0] <= end <= time_slot[1] and start > refr:
                return False
        return True


    def eval_model(self):
        """
        This function overload eval() function of nn.Module
        :return: None
        """
        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False
        # Call the original eval() function
        super(SC_GRU, self).eval()


    def train_model(self, mode=True):
        """
        This function overload train() function of nn.Module
        :param mode: "lstm" - train only lstm and lstm_linear
         layers. It is first step of training
         where only reschedulable tasks are used.
         "injector" - train only injector and freeze other
         layers. It is second step of training
         where only non-reschedulable tasks are used.
        :return: None
        """
        # Set requires_grad to True for all parameters
        for param in self.parameters():
            param.requires_grad = True

        # Call the original train() function
        super(SC_GRU, self).train(True)

        if mode == "lstm":
            self.__train_lstm()

        elif mode == "injector":
            self.__train_injector()


    def __train_lstm(self):
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
        frozen_layers = [self.gru, self.gru_linear, self.lstm_lin_activation]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = True

        self.train_mode = "lstm"


    def __train_injector(self):
        """
        This function sets the model to train only injector
         and freeze other layers. It is second step of training
         where only non-reschedulable tasks are used.
        :return: None
        """
        # To train injector we need to freeze lstm and lstm_linear layers
        frozen_layers = [self.gru, self.gru_linear, self.lstm_lin_activation]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Leave only injector trainable
        for param in self.injector.parameters():
            param.requires_grad = True

        self.train_mode = "injector"


    def reset_states(self):
        """
        This function resets the hidden and cell states of the model to zeros.
        :return: None
        """
        self.hn = torch.zeros(self.gru_layers, self.batch_size, self.hidden)         # GRU intermediate result
        if self.batch_size == 1:
            self.hn = self.hn.squeeze(1)


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


    def set_states(self, hn):
        """
        Sets hidden and cell states of the model.
        :param hn: hidden state
        :param cn: cell state
        :return: None
        """
        # Convert to tensors with appropriate dtype if they are not
        if not isinstance(hn, torch.Tensor):
            hn = torch.tensor(hn).type(torch.float32)

        # Set states
        self.hn = hn