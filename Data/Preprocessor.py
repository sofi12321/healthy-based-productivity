import pandas as pd
import numpy as np
import datetime
import random
from sklearn.preprocessing import MinMaxScaler
from feature_engine.creation import CyclicalFeatures
from tg_bot.domain.domain import Task, Event
from dataclasses import dataclass
from typing import Optional


class Preprocessor:
    def __init__(self):
        pass

    def preprocess_task(self, task: Task, label: int):
        """
        Preprocesses task for model
        :param task: task to preprocess
        :param label: label of task
        :return: vector of input features for model
        """
        input_vector = pd.DataFrame(columns=['Label Number', 'Duration', 'Importance', 'Time_Min',
                                             'Date_Categorical', 'Date_Day', 'Date_Month'])

        label_num = label
        duration = task.duration
        importance = task.importance

        if task.start_time is not None:
            time = task.start_time
            minutes = time.minute + time.hour * 60
        else:
            minutes = 0

        if task.date is not None:
            date = task.date
            day = date.day
            month = date.month
        else:
            date = 0
            day = 0
            month = 0

        input_vector.loc[len(input_vector)] = {'Label Number': label_num,
                                               'Duration': duration,
                                               'Importance': importance,
                                               'Time_Min': minutes,
                                               'Date_Categorical': int(date.strftime("%j")),
                                               'Date_Day': day,
                                               'Date_Month': month}

        cyclical = CyclicalFeatures(variables=['Time_Min', 'Date_Day', 'Date_Month'])
        input_vector = cyclical.fit_transform(input_vector)
        input_vector.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        scaler = MinMaxScaler()
        input_vector['Duration'] = scaler.fit_transform(input_vector[['Duration']])
        input_vector['Time_Min'] = scaler.fit_transform(input_vector[['Time_Min']])

        return input_vector.loc[0].to_numpy()

    def preprocess_event(self, event: Event, label: int):
        """
        Preprocesses event for model
        :param event: event to preprocess
        :param label: label of event
        :return: vector of input features for model
        """
        input_vector = pd.DataFrame(columns=['Label Number', 'Duration', 'Importance', 'Time_Min',
                                             'Date_Categorical', 'Date_Day', 'Date_Month'])

        label_num = label
        duration = event.duration
        importance = 3

        if event.start_time is not None:
            time = event.start_time
            minutes = time.minute + time.hour * 60
        else:
            minutes = 0

        if event.date is not None:
            date = event.date
            day = date.day
            month = date.month
        else:
            date = 0
            day = 0
            month = 0

        input_vector.loc[len(input_vector)] = {'Label Number': label_num,
                                               'Duration': duration,
                                               'Importance': importance,
                                               'Time_Min': minutes,
                                               'Date_Categorical': int(date.strftime("%j")),
                                               'Date_Day': day,
                                               'Date_Month': month}

        cyclical = CyclicalFeatures(variables=['Time_Min', 'Date_Day', 'Date_Month'])
        input_vector = cyclical.fit_transform(input_vector)
        input_vector.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        scaler = MinMaxScaler()
        input_vector['Duration'] = scaler.fit_transform(input_vector[['Duration']])
        input_vector['Time_Min'] = scaler.fit_transform(input_vector[['Time_Min']])

        return input_vector.loc[0].to_numpy()


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     task = Task(telegram_id=1, task_name='test', duration=30, importance=1, start_time=datetime.time(12, 30),
#                 date=datetime.date(2021, 5, 1))
#     event = Event(telegram_id=1, event_name='test', duration=30, start_time=datetime.time(12, 30),
#                   date=datetime.date(2021, 5, 1))
#     preprocessor = Preprocessor()
#     input_vector = preprocessor.preprocess_task(task, 0)
#     print(input_vector)
#     input_vector = preprocessor.preprocess_event(event, 0)
#     print(input_vector)

