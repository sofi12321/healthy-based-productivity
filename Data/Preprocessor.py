import pandas as pd
import numpy as np
import datetime
from tg_bot.domain.domain import Task, Event
import calendar


class Preprocessor:
    def __init__(self):
        """
        Preprocessor for generated data
        """

    def _make_dataframe(self):
        """
        Creates dataframe with columns for model
        :return: dataframe with columns for model
        """
        df = pd.DataFrame(
            columns=['Label Number',
                     'Duration',
                     'Importance',
                     'Time_Min',
                     'Date_Categorical',
                     'Date_Day',
                     'Date_Month'
                     ]
        )
        return df

    def _encode(self, data, column_name, num_labels=4):
        """
        Encodes feature with one-hot encoding
        :param data: dataframe with feature to encode
        :param column_name: name of column to encode
        :param num_labels: number of labels
        :return: dataframe with encoded feature
        """
        encoded_labels = [f"{column_name}_{i}" for i in range(num_labels)]
        label_num = data[f"{column_name}"].to_numpy().reshape(-1, 1)
        data.drop(columns=[f"{column_name}"], inplace=True)

        for j in range(len(label_num)):
            for i in range(num_labels):
                data.insert(j, encoded_labels[i], 1 if i == label_num[j] else 0)
        return data

    def _transform_cyclical_features(self, data, start_date):
        """
        Transforms cyclical features to sin and cos
        :param data: dataframe with cyclical features
        :return: dataframe with transformed features
        """
        transformed_data = data.copy()

        # Get the number of days in the month of the provided start date of the activity
        if start_date == 0:
            start_date = datetime.datetime.now()
        num_days_start = calendar.monthrange(start_date.year, start_date.month)[1]

        # Transform Time_Min feature
        transformed_data['Time_Min_sin'] = np.sin(2 * np.pi * data['Time_Min'] / 1440)
        transformed_data['Time_Min_cos'] = np.cos(2 * np.pi * data['Time_Min'] / 1440)

        # Transform Date_Day feature
        transformed_data['Date_Day_sin'] = np.sin(2 * np.pi * data['Date_Day'] / num_days_start)
        transformed_data['Date_Day_cos'] = np.cos(2 * np.pi * data['Date_Day'] / num_days_start)

        # Transform Date_Month feature
        transformed_data['Date_Month_sin'] = np.sin(2 * np.pi * data['Date_Month'] / 12)
        transformed_data['Date_Month_cos'] = np.cos(2 * np.pi * data['Date_Month'] / 12)

        # Drop old features except Time_Min because it is used as feature for model
        transformed_data.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        return transformed_data

    def preprocess_task(self, task: Task, label: int, plan_time):
        """
        Preprocesses task for model
        :param task: task to preprocess
        :param label: label of task
        :param plan_time: time when task was planned
        :return: vector of input features for model
        """
        input_vector = self._make_dataframe()

        # Parse task object to input vector
        label_num = label
        duration = task.duration
        importance = task.importance

        if task.start_time is not None:
            start_time = task.start_time
            start_minutes = start_time.minute + start_time.hour * 60
        else:
            start_minutes = 0

        if task.date is not None:
            start_date = task.date
            start_day = start_date.day
            start_month = start_date.month
        else:
            start_date = 0
            start_day = 0
            start_month = 0

        input_vector.loc[len(input_vector)] = {'Label Number': label_num,
                                               'Duration': duration,
                                               'Importance': importance,
                                               'Time_Min': start_minutes,
                                               'Date_Categorical': int(start_date.strftime("%j")),
                                               'Date_Day': start_day,
                                               'Date_Month': start_month
                                               }

        return self.preprocess_activity(input_vector, start_date, plan_time)

    def preprocess_event(self, event: Event, label: int, plan_time):
        """
        Preprocesses event for model
        :param event: event to preprocess
        :param label: label of event
        :param plan_time: time when event was planned
        :return: vector of input features for model
        """
        input_vector = self._make_dataframe()

        # Parse event object to input vector
        label_num = label
        duration = event.duration
        importance = 3

        if event.start_time is not None:
            start_time = event.start_time
            start_minutes = start_time.minute + start_time.hour * 60
        else:
            start_minutes = 0

        if event.date is not None:
            start_date = event.date
            start_day = start_date.day
            start_month = start_date.month
        else:
            start_date = 0
            start_day = 0
            start_month = 0

        input_vector.loc[len(input_vector)] = {'Label Number': label_num,
                                               'Duration': duration,
                                               'Importance': importance,
                                               'Time_Min': start_minutes,
                                               'Date_Categorical': int(start_date.strftime("%j")),
                                               'Date_Day': start_day,
                                               'Date_Month': start_month
                                               }

        return self.preprocess_activity(input_vector, start_date, plan_time)

    def preprocess_activity(self, input_vector, start_date, plan_time):
        # If Time is not in minutes, convert it to minutes
        if not isinstance(input_vector['Time_Min'][0], np.int64):
            max_time = plan_time + datetime.timedelta(minutes=1440)
            temp_time = input_vector['Time_Min'] * (max_time - plan_time) + plan_time
            input_vector['Time_Min'] = temp_time.dt.hour * 60 + temp_time.dt.minute

        # Convert Time_Min to sin and cos# Transform cyclical features
        input_vector = self._transform_cyclical_features(input_vector, start_date)

        # convert Time_Min to alpha format
        max_time = plan_time + datetime.timedelta(minutes=1440)
        start_date = datetime.datetime(start_date.year, start_date.month, start_date.day,
                                       input_vector['Time_Min'][0] // 60, input_vector['Time_Min'][0] % 60)
        input_vector['Time_Min'] = (start_date - plan_time) / (max_time - plan_time)

        # Encode label number
        input_vector = self._encode(input_vector, "Label Number")

        # Scale Duration assuming that duration does not exceed 240 minutes
        input_vector['Duration'] = input_vector['Duration'] / 240

        # Scale Date_Categorical and Importance
        input_vector['Date_Categorical'] = input_vector['Date_Categorical'] / 365
        input_vector['Importance'] = input_vector['Importance'] / 3

        # Rearrange columns of input vector
        input_vector = input_vector[['Label Number_0', 'Label Number_1', 'Label Number_2', 'Label Number_3',
                                     'Duration',
                                     'Importance',
                                     'Time_Min', 'Time_Min_sin', 'Time_Min_cos',
                                     'Date_Categorical',
                                     'Date_Day_sin', 'Date_Day_cos',
                                     'Date_Month_sin', 'Date_Month_cos']]

        return input_vector.loc[0].to_numpy()


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     task = Task(telegram_id=1, task_name='test', duration=30, importance=1, start_time=datetime.time(12, 30),
#                 date=datetime.date(2021, 5, 1))
#     event = Event(telegram_id=1, event_name='test', duration=30, start_time=datetime.time(12, 30),
#                   date=datetime.date(2021, 5, 1))
#     preprocessor = Preprocessor()
#     input_vector = preprocessor.preprocess_task(task, 0, datetime.datetime(2021, 5, 1, 12, 0))
#     print(input_vector)
#     input_vector = preprocessor.preprocess_event(event, 0, datetime.datetime(2021, 5, 1, 12, 0))
#     print(input_vector)
