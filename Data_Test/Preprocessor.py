import numpy as np
import datetime

class Preprocessor:
    def __init__(self, alpha, num_lables, beta, max_importance):
        self.alpha = alpha
        self.beta = beta
        self.num_labels = num_lables
        self.max_importance = max_importance
        self.features_num = self.num_labels + 1 + 1 + 7 + 1 + 7 + 1


    def _convert_time_to_alpha(self, time):
        return (time.hour * 60 + time.minute) / self.alpha



    def preprocess_data(self, task_type, data, label):
        next_vec_iter = 0

        # TODO: Dynamic number of features
        model_in_vector = np.zeros(self.features_num, dtype=np.float32)

        # Label
        model_in_vector[label] = 1
        next_vec_iter += self.num_labels

        # Duration
        duration = data.duration / self.beta
        model_in_vector[next_vec_iter] = duration
        next_vec_iter += 1

        # Current Time
        current_time = data.curr_time
        current_time = self._convert_time_to_alpha(current_time)
        model_in_vector[next_vec_iter] = current_time
        next_vec_iter += 1

        # Current Week Day
        current_date = data.curr_date
        model_in_vector[next_vec_iter + current_date.weekday()] = 1
        next_vec_iter += 7

        # Start Time
        start_time = self._convert_time_to_alpha(data.start_time)
        model_in_vector[next_vec_iter] = start_time
        next_vec_iter += 1

        # Start Week Day
        day = current_date.weekday()
        if start_time < current_time:
            # Get next week day
            day = (current_date + datetime.timedelta(days=1)).weekday()

        model_in_vector[next_vec_iter + day] = 1
        next_vec_iter += 7

        if task_type == "non-resched":
            # Importance
            model_in_vector[next_vec_iter] = 1


        elif task_type == "resched":
            # Importance
            importance = data.importance / self.max_importance
            model_in_vector[next_vec_iter] = importance
            next_vec_iter += 1

        return model_in_vector