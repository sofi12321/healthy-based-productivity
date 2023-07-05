import pandas as pd
import numpy as np
import datetime
import random
from sklearn.preprocessing import MinMaxScaler
from feature_engine.creation import CyclicalFeatures
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    task_name: str
    duration: int
    importance: int
    complexity: int
    start_time: Optional[datetime.time] = None
    date: Optional[datetime.date] = None


@dataclass
class BasicUserInfo:
    start_time: datetime.time
    end_time: datetime.time


@dataclass
class Event:
    event_name: str
    start_time: datetime.time
    duration: int
    repeat_arguments: Optional[str] = None


class Preprocessor:
    def __init__(self):
        pass

    # there should be NLP))
    def _assign_label_number(self, label):
        labels = {
            "Sport": 0,
            "Food": 1,
            "Hobby Active": 2,
            "Hobby Passive": 3,
            "Studying": 4,
            "Work": 5,
            "Other": 6
        }
        return labels[label]

    def _get_nearest_day(self, day_name):
        # Get today's date
        today = datetime.date.today()

        # Map day names to weekday numbers
        weekdays = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6
        }

        # Get the weekday number of the given day_name
        target_weekday = weekdays[day_name.lower()]

        # Calculate the difference between today's weekday and the target weekday
        days_ahead = (target_weekday - today.weekday()) % 7

        # Calculate the date of the nearest day
        nearest_day = today + datetime.timedelta(days=days_ahead)

        return nearest_day


    def _parse_queue(self, queue):
        input = pd.DataFrame(columns=['Label Number', 'Duration', 'Importance', 'Time_Min',
                                      'Date_Day', 'Date_Month', 'Type'])

        for i in range(len(queue)):
            if isinstance(queue[i], Task):
                task = queue[i]
                label_num = self._assign_label_number(task.task_name)
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
                    day = 0
                    month = 0

                input.loc[len(input)] = {'Label Number': label_num,
                                         'Duration': duration,
                                         'Importance': importance,
                                         'Time_Min': minutes,
                                         'Date_Day': day, 'Date_Month': month,
                                         'Type': 'resched'
                                         }
            else:
                event = queue[i]
                label_num = self._assign_label_number(event.event_name)
                duration = event.duration
                importance = 3

                if event.start_time is not None:
                    time = event.start_time
                    minutes = time.minute + time.hour * 60
                else:
                    minutes = 0

                if event.repeat_arguments is not None:
                    repeat_arguments = event.repeat_arguments.split()
                    shift = int(repeat_arguments[0])
                    day_name = repeat_arguments[1]
                    repeat = int(repeat_arguments[2])
                    for i in range(int(repeat_arguments[2])):
                        date = self._get_nearest_day(day_name) + datetime.timedelta(days=shift * i * 7)
                        # print(date)
                        day = date.day
                        month = date.month
                        input.loc[len(input)] = {'Label Number': label_num,
                                                 'Duration': duration,
                                                 'Importance': importance,
                                                 'Time_Min': minutes,
                                                 'Date_Day': day, 'Date_Month': month,
                                                 'Type': 'non-resched'
                                                 }
                else:
                    date = datetime.date.today()
                    day = date.day
                    month = date.month
                    input.loc[len(input)] = {'Label Number': label_num,
                                             'Duration': duration,
                                             'Importance': importance,
                                             'Time_Min': minutes,
                                             'Date_Day': day, 'Date_Month': month,
                                             'Type': 'non-resched'
                                             }

        input.sort_values(by=['Importance'], inplace=True, ascending=False)
        input.reset_index(drop=True, inplace=True)

        task_type = input['Type'].to_numpy()
        input.drop(columns=['Type'], inplace=True)

        output = np.empty(len(input), dtype=object)
        for i in range(len(input)):
            if task_type[i] == 'non-resched':
                output[i] = np.array([input['Time_Min'][i], input['Duration'][i], 0])
            else:
                output[i] = np.nan

        return input, task_type, output

    def preprocess(self, queue):

        if not isinstance(queue, pd.DataFrame):
            input_vector, type_vector, output_vector = self._parse_queue(queue)
        else:
            input_vector = queue
            input_vector["Start Time"] = pd.to_datetime(input_vector["Start Time"], format="%H:%M")
            input_vector["Time_Min"] = input_vector["Start Time"].dt.minute + input_vector["Start Time"].dt.hour * 60
            input_vector["Date"] = pd.to_datetime(input_vector["Date"], format="%d/%m/%Y")
            input_vector["Date_Day"] = input_vector["Date"].dt.day
            input_vector["Date_Month"] = input_vector["Date"].dt.month
            input_vector["Date"] = input_vector["Date"].dt.strftime("%j").astype(int)
            input_vector.drop(columns=["Start Time"], inplace=True)

            type_vector = np.empty(len(input_vector), dtype=object)
            for i in range(len(type_vector)):
                type_vector[i] = np.random.choice(["resched", "non-resched"], p=[0.8, 0.2])

            output_vector = np.empty((len(input_vector), 3), dtype=object)
            for i in range(len(input_vector)):
                if type_vector[i] == 'non-resched':
                    output_vector[i][0] = input_vector['Time_Min'][i]
                    output_vector[i][1] = input_vector['Duration'][i]
                    output_vector[i][2] = 0
                else:
                    if random.random() > 0.8:
                        shift = random.randint(-15, 15)
                        output_vector[i][0] = input_vector['Time_Min'][i]
                        output_vector[i][1] = input_vector['Duration'][i] + shift
                        output_vector[i][2] = shift
                    else:
                        output_vector[i][0] = input_vector['Time_Min'][i]
                        output_vector[i][1] = input_vector['Duration'][i]
                        output_vector[i][2] = 0

        cyclical = CyclicalFeatures(variables=['Time_Min', 'Date_Day', 'Date_Month'])
        input_vector = cyclical.fit_transform(input_vector)
        input_vector.drop(columns=['Date_Day', 'Date_Month'], inplace=True)

        scaler = MinMaxScaler()
        input_vector['Duration'] = scaler.fit_transform(input_vector[['Duration']])
        input_vector['Time_Min'] = scaler.fit_transform(input_vector[['Time_Min']])

        # Convert the output_vector to the same format as an input_vector
        output_vector = pd.DataFrame(output_vector, columns=['start', 'end', 'refr'])

        return input_vector, type_vector, output_vector

# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     # queue = [Task(task_name="Sport", duration=30, importance=3, start_time=datetime.time(hour=8, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=1),
#     #          Task(task_name="Food", duration=60, importance=2, start_time=datetime.time(hour=12, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=1),
#     #          Task(task_name="Hobby Active", duration=120, importance=1, start_time=datetime.time(hour=16, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=2),
#     #          Task(task_name="Hobby Passive", duration=60, importance=1, start_time=datetime.time(hour=18, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=0),
#     #          Task(task_name="Studying", duration=120, importance=3, start_time=datetime.time(hour=20, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=3),
#     #          Task(task_name="Work", duration=120, importance=3, start_time=datetime.time(hour=22, minute=0), date=datetime.date(year=2021, month=5, day=1), complexity=3),
#     #          Task(task_name="Other", duration=120, importance=3, start_time=datetime.time(hour=0, minute=0), date=datetime.date(year=2021, month=5, day=2), complexity=0),
#     #          Event(event_name="Sport", duration=120, repeat_arguments="1 monday 6", start_time=datetime.time(hour=0, minute=0)),
#     #          Event(event_name="Sport", duration=120, start_time=datetime.time(hour=0, minute=0))]
#
#     queue = pd.read_csv("schedule_v3.csv")
#
#     preprocessor = Preprocessor()
#     input_vector, type_vector, output_vector = preprocessor.preprocess(queue)
#     print(input_vector)
#     print(type_vector)
#     print(output_vector)
