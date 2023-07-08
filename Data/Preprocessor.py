import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tg_bot.domain.domain import Task, Event
import calendar


class Preprocessor:
    def __init__(self):
        """
        Preprocessor for generated data
            - duration_scaler: scaler for duration feature
        """
        self.duration_scaler = MinMaxScaler()

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
                     'Date_Month',
                     'Plan_Time_Min',
                     'Plan_Date_Categorical',
                     'Plan_Date_Day',
                     'Plan_Date_Month'
                     ]
        )
        return df

    def _encode_label(self, data, num_labels=4):
        """
        Encodes feature
        :param data: dataframe with feature to encode
        :param column_name: name of column to encode
        :param num_labels: number of labels
        :return: dataframe with encoded feature
        """
        encoded_labels = [f"Label Number_{i}" for i in range(num_labels)]
        label_num = data['Label Number'].to_numpy().reshape(-1, 1)
        data.drop(columns=['Label Number'], inplace=True)

        for j in range(len(label_num)):
            for i in range(num_labels):
                data.insert(i, encoded_labels[i], 1 if i == label_num[j] else 0)
        return data

    def _transform_cyclical_features(self, data, start_date, plan_date):
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

        # Get the number of days in the month of the provided date when the activity was planned by user
        num_days_plan = calendar.monthrange(plan_date.year, plan_date.month)[1]

        # Transform Time_Min feature
        transformed_data['Time_Min_sin'] = np.sin(2 * np.pi * data['Time_Min'] / 1440)
        transformed_data['Time_Min_cos'] = np.cos(2 * np.pi * data['Time_Min'] / 1440)

        # Transform Date_Day feature
        transformed_data['Date_Day_sin'] = np.sin(2 * np.pi * data['Date_Day'] / num_days_start)
        transformed_data['Date_Day_cos'] = np.cos(2 * np.pi * data['Date_Day'] / num_days_start)

        # Transform Date_Month feature
        transformed_data['Date_Month_sin'] = np.sin(2 * np.pi * data['Date_Month'] / 12)
        transformed_data['Date_Month_cos'] = np.cos(2 * np.pi * data['Date_Month'] / 12)

        # Transform Plan_Time_Min feature
        transformed_data['Plan_Time_Min_sin'] = np.sin(2 * np.pi * data['Plan_Time_Min'] / 1440)
        transformed_data['Plan_Time_Min_cos'] = np.cos(2 * np.pi * data['Plan_Time_Min'] / 1440)

        # Transform Plan_Date_Day feature
        transformed_data['Plan_Date_Day_sin'] = np.sin(2 * np.pi * data['Plan_Date_Day'] / num_days_plan)
        transformed_data['Plan_Date_Day_cos'] = np.cos(2 * np.pi * data['Plan_Date_Day'] / num_days_plan)

        # Transform Plan_Date_Month feature
        transformed_data['Plan_Date_Month_sin'] = np.sin(2 * np.pi * data['Plan_Date_Month'] / 12)
        transformed_data['Plan_Date_Month_cos'] = np.cos(2 * np.pi * data['Plan_Date_Month'] / 12)

        # Drop old features except Time_Min because it is used as feature for model
        transformed_data.drop(columns=['Date_Day', 'Date_Month', 'Plan_Date_Day', 'Plan_Date_Month'], inplace=True)

        return transformed_data

    def preprocess_task(self, task: Task, label: int, plan_time):
        """
        Preprocesses task for model
        :param task: task to preprocess
        :param label: label of task
        :return: vector of input features for model
        """
        input_vector = self._make_dataframe()

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
                                               'Date_Month': start_month,
                                               'Plan_Time_Min': plan_time.minute + plan_time.hour * 60,
                                               'Plan_Date_Categorical': int(plan_time.strftime("%j")),
                                               'Plan_Date_Day': plan_time.day,
                                               'Plan_Date_Month': plan_time.month}

        return self._preprocess_activity(input_vector, start_date, plan_time)

    def preprocess_event(self, event: Event, label: int, plan_time):
        """
        Preprocesses event for model
        :param event: event to preprocess
        :param label: label of event
        :return: vector of input features for model
        """
        input_vector = self._make_dataframe()

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
                                               'Date_Month': start_month,
                                               'Plan_Time_Min': plan_time.minute + plan_time.hour * 60,
                                               'Plan_Date_Categorical': int(plan_time.strftime("%j")),
                                               'Plan_Date_Day': plan_time.day,
                                               'Plan_Date_Month': plan_time.month}

        return self._preprocess_activity(input_vector, start_date, plan_time)

    def _preprocess_activity(self, input_vector, start_date, plan_time):
        # Encode label number
        input_vector = self._encode_label(input_vector)

        # Transform cyclical features
        input_vector = self._transform_cyclical_features(input_vector, start_date, plan_time)

        # Update scaler for Duration and then scale Duration
        self.duration_scaler.partial_fit(input_vector['Duration'].values.reshape(-1, 1))
        input_vector['Duration'] = self.duration_scaler.transform(input_vector['Duration'].values.reshape(-1, 1))

        # Scale Time_Min, Date_Categorical, Plan_Time_Min, Plan_Date_Categorical
        input_vector['Time_Min'] = input_vector['Time_Min'] / 1440
        input_vector['Date_Categorical'] = input_vector['Date_Categorical'] / 365
        input_vector['Plan_Time_Min'] = input_vector['Plan_Time_Min'] / 1440
        input_vector['Plan_Date_Categorical'] = input_vector['Plan_Date_Categorical'] / 365

        # Rearrange columns of input vector
        input_vector = input_vector[['Label Number_0', 'Label Number_1', 'Label Number_2', 'Label Number_3',
                                     'Duration',
                                     'Importance',
                                     'Time_Min', 'Time_Min_sin', 'Time_Min_cos',
                                     'Date_Categorical',
                                     'Date_Day_sin', 'Date_Day_cos',
                                     'Date_Month_sin', 'Date_Month_cos',
                                     'Plan_Time_Min', 'Plan_Time_Min_sin', 'Plan_Time_Min_cos',
                                     'Plan_Date_Categorical',
                                     'Plan_Date_Day_sin', 'Plan_Date_Day_cos',
                                     'Plan_Date_Month_sin', 'Plan_Date_Month_cos']]

        return input_vector.loc[0].to_numpy()


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     task = Task(telegram_id=1, task_name='test', duration=30, importance=1, start_time=datetime.time(12, 30),
#                 date=datetime.date(2021, 5, 1))
#     event = Event(telegram_id=1, event_name='test', duration=30, start_time=datetime.time(12, 30),
#                   date=datetime.date(2021, 5, 1))
#     preprocessor = Preprocessor()
#     input_vector = preprocessor.preprocess_task(task, 0, datetime.datetime(2021, 5, 1, 12, 30))
#     print(input_vector)
#     input_vector = preprocessor.preprocess_event(event, 0, datetime.datetime(2021, 5, 1, 12, 30))
#     print(input_vector)
