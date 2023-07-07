import pandas as pd
import numpy as np
import datetime
import random
from sklearn.preprocessing import MinMaxScaler
from feature_engine.creation import CyclicalFeatures
from dataclasses import dataclass
from typing import Optional
from tg_bot.domain.domain import Task, Event
from Data.converter import Converter


class Preprocessor:
    def __init__(self):
        """
        Preprocessor for generated data
        """
        self.converter = Converter(alpha=1440)

    def preprocess(self, filename="schedule_gen.csv"):
        """
        Preprocesses generated data for model
        :param filename: name of file with generated data
        :return:
            - vector of input features for model
            - vector describing type of activity (resched or non-resched)
            - vector of expected output of model
            - vector of time when the task/event was planned by user
        """
        input_vector = pd.read_csv(filename)

        # Modify input vector
        input_vector["Start Time"] = pd.to_datetime(input_vector["Start Time"], format="%H:%M")
        input_vector["Time_Min"] = input_vector["Start Time"].dt.minute + input_vector["Start Time"].dt.hour * 60
        input_vector["Date"] = pd.to_datetime(input_vector["Date"], format="%d/%m/%Y")
        input_vector["Date_Categorical"] = input_vector["Date"].dt.strftime("%j").astype(int)
        input_vector["Date_Day"] = input_vector["Date"].dt.day
        input_vector["Date_Month"] = input_vector["Date"].dt.month

        # Fill type vector
        type_vector = np.empty(len(input_vector), dtype=object)
        for i in range(len(type_vector)):
            type_vector[i] = np.random.choice(["resched", "non-resched"], p=[0.8, 0.2])

        # Fill output vector
        output_vector = np.empty((len(input_vector), 3), dtype=object)
        for i in range(len(input_vector)):
            # Convert output vector if type is non-resched
            if type_vector[i] == 'non-resched':
                output_vector[i] = self.converter.user_to_model(
                    task_date=datetime.datetime(
                        year=input_vector['Date'][i].year,
                        month=input_vector['Date'][i].month,
                        day=input_vector['Date'][i].day,
                        hour=input_vector['Time_Min'][i] // 60,
                        minute=input_vector['Time_Min'][i] % 60,
                    ),
                    duration=int(input_vector['Duration'][i]),
                    offset=0
                )
            else:
                # Shift time and duration with some probability
                if random.random() > 0.8:
                    shift = random.randint(-15, 15)
                    output_vector[i][0] = input_vector['Time_Min'][i]
                    output_vector[i][1] = input_vector['Duration'][i] + shift
                    output_vector[i][2] = shift
                else:
                    output_vector[i][0] = input_vector['Time_Min'][i]
                    output_vector[i][1] = input_vector['Duration'][i]
                    output_vector[i][2] = 0

                # Delete date with some probability for augmentation
                if random.random() > 0.8:
                    input_vector.loc[i, "Date_Categorical"] = 0
                    input_vector.loc[i, "Date_Month"] = 0
                    input_vector.loc[i, "Date_Day"] = 0

                # Delete time with some probability for augmentation
                if random.random() > 0.8:
                    input_vector.loc[i, "Time_Min"] = 0

        # Fill plan time vector
        plan_time_vector = np.empty(len(input_vector), dtype=object)
        temp_day = input_vector['Date_Day'][0]
        temp_time = input_vector['Time_Min'][0] - random.randint(3, 7)
        temp_time = 0 if temp_time < 0 else temp_time
        for i in range(len(plan_time_vector)):
            # If new day, then set new temp data
            if temp_day != input_vector['Date_Day'][i]:
                temp_day = input_vector['Date_Day'][i]
                temp_time = input_vector['Time_Min'][i] - random.randint(3, 7)
                temp_time = 0 if temp_time < 0 else temp_time

            # Set plan time in the beginning of the day
            if random.random() < 0.9:
                time = temp_time
            # Set plan time right before the activity
            else:
                time = input_vector['Time_Min'][i] - random.randint(5, 15)
                time = 0 if time < 0 else time
            plan_time_vector[i] = datetime.datetime(
                year=input_vector['Date'][i].year,
                month=input_vector['Date'][i].month,
                day=input_vector['Date'][i].day,
                hour=time // 60,
                minute=time % 60,
            )

        # Drop unnecessary columns
        input_vector.drop(columns=['Date', 'Start Time'], inplace=True)

        # Convert cyclical features
        cyclical = CyclicalFeatures(variables=['Time_Min', 'Date_Day', 'Date_Month'])
        input_vector = cyclical.fit_transform(input_vector)
        input_vector.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        # Scale duration and time
        scaler = MinMaxScaler()
        input_vector['Duration'] = scaler.fit_transform(input_vector[['Duration']])
        input_vector['Time_Min'] = scaler.fit_transform(input_vector[['Time_Min']])

        # Convert the output_vector to the same format as an input_vector
        output_vector = pd.DataFrame(output_vector, columns=['start', 'end', 'refr'])

        return input_vector, type_vector, output_vector, plan_time_vector


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#
#     preprocessor = Preprocessor()
#     input_vector, type_vector, output_vector, plan_time_vector = preprocessor.preprocess("schedule_gen.csv")
#     print(input_vector)
#     print(type_vector)
#     print(output_vector)
#     print(plan_time_vector)

#%%
