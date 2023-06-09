import pandas as pd
import numpy as np
import datetime
import random
from Data.converter import Converter
import calendar


class Preprocessor:
    def __init__(self):
        """
        Preprocessor for generated data
        """
        self.converter = Converter(alpha=1440)

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

        for j in range(num_labels):
            data.insert(j, encoded_labels[j], 0)

        for j in range(len(label_num)):
            for i in range(num_labels):
                data.loc[j, encoded_labels[i]] = 1 if i == label_num[j] else 0
        return data

    def _transform_cyclical_features(self, data):
        """
        Transforms cyclical features to sin and cos
        :param data: dataframe with cyclical features
        :return: dataframe with transformed features
        """
        transformed_data = data.copy()

        # Add new columns for transformed features
        new_columns = [
            'Time_Min_sin', 'Time_Min_cos', 'Date_Day_sin', 'Date_Day_cos', 'Date_Month_sin', 'Date_Month_cos']
        for column in new_columns:
            transformed_data[column] = 0

        for i in range(len(data)):
            start_date = data['Date'].loc[i]
            plan_date = data['Plan_Date'].loc[i]

            # Get the number of days in the month of the provided start date of the activity
            if start_date == 0:
                start_date = datetime.datetime.now()
            num_days_start = calendar.monthrange(start_date.year, start_date.month)[1]

            # Transform Time_Min feature
            transformed_data.loc[i, 'Time_Min_sin'] = np.sin(2 * np.pi * data['Time_Min'][i] / 1440)
            transformed_data.loc[i, 'Time_Min_cos'] = np.cos(2 * np.pi * data['Time_Min'][i] / 1440)

            # Transform Date_Day feature
            transformed_data.loc[i, 'Date_Day_sin'] = np.sin(2 * np.pi * data['Date_Day'][i] / num_days_start)
            transformed_data.loc[i, 'Date_Day_cos'] = np.cos(2 * np.pi * data['Date_Day'][i] / num_days_start)

            # Transform Date_Month feature
            transformed_data.loc[i, 'Date_Month_sin'] = np.sin(2 * np.pi * data['Date_Month'][i] / 12)
            transformed_data.loc[i, 'Date_Month_cos'] = np.cos(2 * np.pi * data['Date_Month'][i] / 12)

        # Drop old features except Time_Min because it is used as feature for model
        transformed_data.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        return transformed_data

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

        # Fill plan time
        temp_day = input_vector['Date_Day'][0]
        temp_time = input_vector['Time_Min'][0] - random.randint(3, 7)
        temp_time = 0 if temp_time < 0 else temp_time
        for i in range(len(input_vector)):
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

            input_vector.loc[i, "Plan_Date"] = datetime.datetime(
                year=input_vector['Date'][i].year,
                month=input_vector['Date'][i].month,
                day=input_vector['Date'][i].day,
                hour=time // 60,
                minute=time % 60,
            )

        # Fill output vector
        output_vector = np.empty((len(input_vector), 3), dtype=object)
        for i in range(len(input_vector)):
            shift = 0
            if type_vector[i] == 'resched':
                # Shift time and duration with some probability
                if random.random() > 0.8:
                    shift = random.randint(-15, 15)

                # Delete date with some probability for augmentation
                if random.random() > 0.8:
                    input_vector.loc[i, "Date_Categorical"] = 0
                    input_vector.loc[i, "Date_Month"] = 0
                    input_vector.loc[i, "Date_Day"] = 0

                # Delete time with some probability for augmentation
                if random.random() > 0.8:
                    input_vector.loc[i, "Time_Min"] = 0

            # Convert output vector
            output_vector[i] = self.converter.user_to_model(
                task_date=datetime.datetime(
                    year=input_vector['Date'][i].year,
                    month=input_vector['Date'][i].month,
                    day=input_vector['Date'][i].day,
                    hour=input_vector['Time_Min'][i] // 60,
                    minute=input_vector['Time_Min'][i] % 60
                ),
                duration=int(input_vector['Duration'][i]) + shift,
                offset=shift,
                current_date=input_vector['Plan_Date'][i]
            )

        # Transform cyclical features
        input_vector = self._transform_cyclical_features(input_vector)

        # convert Time_Min to alpha format
        for i in range(len(input_vector)):
            current_date = input_vector['Plan_Date'][i]
            max_time = current_date + datetime.timedelta(minutes=1440)
            task_date = datetime.datetime(
                year=input_vector['Date'][i].year,
                month=input_vector['Date'][i].month,
                day=input_vector['Date'][i].day,
                hour=int(input_vector['Time_Min'][i]) // 60,
                minute=int(input_vector['Time_Min'][i]) % 60
            )
            input_vector.loc[i, 'Time_Min'] = (task_date - current_date) / (max_time - current_date)

        # Encode label number
        input_vector = self._encode(input_vector, "Label Number")

        # Scale Duration assuming that duration does not exceed 240 minutes
        input_vector['Duration'] = input_vector['Duration'] / 240

        # Scale Time_Min and Importance
        input_vector['Date_Categorical'] = input_vector['Date_Categorical'] / 365
        input_vector['Importance'] = input_vector['Importance'] / 3

        # Drop unnecessary columns
        input_vector.drop(columns=['Date', 'Start Time', 'Plan_Date'], inplace=True)

        # Rearrange columns of input vector
        input_vector = input_vector[['Label Number_0', 'Label Number_1', 'Label Number_2', 'Label Number_3',
                                     'Duration',
                                     'Importance',
                                     'Time_Min', 'Time_Min_sin', 'Time_Min_cos',
                                     'Date_Categorical',
                                     'Date_Day_sin', 'Date_Day_cos',
                                     'Date_Month_sin', 'Date_Month_cos']]

        # Convert the output_vector to the same format as an input_vector
        output_vector = pd.DataFrame(output_vector, columns=['start', 'duration', 'refr'])
        if isinstance(output_vector["duration"][0], tuple):
            output_vector["duration"] = output_vector["duration"].apply(lambda x: x[0])

        return input_vector, type_vector, output_vector


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#
#     preprocessor = Preprocessor()
#     input_vector, type_vector, output_vector = preprocessor.preprocess("schedule_gen.csv")
#     print(input_vector.columns)
#     print(input_vector["Time_Min"])
#     print(type_vector)
#     print(output_vector)
