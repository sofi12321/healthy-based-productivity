import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from feature_engine.creation import CyclicalFeatures


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



    # change queue of tasks to dataframe
    # it is assumed that input format is like "Sport-90-3-1-9:00-1/1/1970", two last arguments are optional
    def _tasks_to_df(self, tasks):
        df = pd.DataFrame(columns=['Label Number', 'Duration', 'Importance', 'Time_Min',
                                   'Date_Day', 'Date_Month'])

        for task in tasks:
            task = task.split('-')

            label_num = self._assign_label_number(task[0])
            duration = task[1]
            importance = task[2]

            if len(task) == 6:
                time = datetime.strptime(task[4], '%H:%M')
                date = datetime.strptime(task[5], '%d/%m/%Y')
                minutes = time.minute + time.hour * 60
                day = date.day
                month = date.month
            elif len(task) == 5:
                if task[4].find('/') != -1:
                    date = datetime.strptime(task[4], '%d/%m/%Y')
                    minutes = 0
                    day = date.day
                    month = date.month
                else:
                    time = datetime.strptime(task[4], '%H:%M')
                    minutes = time.minute + time.hour * 60
                    day = 0
                    month = 0
            else:
                minutes = 0
                day = 0
                month = 0

            df.loc[len(df)] = {'Label Number': label_num,
                               'Duration': duration,
                               'Importance': importance,
                               'Time_Min': minutes,
                               'Date_Day': day, 'Date_Month': month
                               }

        df.sort_values(by=['Importance'], inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)

        return df



    # if input format "AML-11:00-90-2 Sunday 3"  or  "AML-11:00-90-2 31 3"  or  "AML-11:00-90-1/1/1970"
    def _events_to_arr_of_dict(self, events):
        arr_of_dict = []
        return arr_of_dict



    def preprocess(self, data):

        if not isinstance(data, pd.DataFrame):
            # temporary solution for tasks/events distinction
            if len(data[0].split("-")) > 4:
                data = self._tasks_to_df(data)
            else:
                data = self._events_to_arr_of_dict(data)
                # special case to skip the neural network
                pass
        else:
            data["Start Time"] = pd.to_datetime(data["Start Time"], format="%H:%M")
            data["Time_Min"] = data["Start Time"].dt.minute + data["Start Time"].dt.hour * 60

            data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
            data["Date_Day"] = data["Date"].dt.day
            data["Date_Month"] = data["Date"].dt.month

            data.drop(columns=["Start Time", "Date"], inplace=True)

        scaler = MinMaxScaler()
        data['Duration'] = scaler.fit_transform(data[['Duration']])

        cyclical = CyclicalFeatures(variables=['Time_Min', 'Date_Day', 'Date_Month'], drop_original=True, )
        data = cyclical.fit_transform(data)

        return data