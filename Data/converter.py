from datetime import datetime, timedelta
import pandas as pd
from Data.Preprocessor import Preprocessor
import torch
import numpy as np


class Converter:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def model_to_user(self, time, duration, offset,
                      current_date=datetime.now().replace(second=0, microsecond=0)):

        # set time to task date
        hours = time * self.alpha // 60
        minutes = round(time * self.alpha % 60)
        task_date_user = current_date.replace(hour=hours, minute=minutes)

        if task_date_user < current_date:
            task_date_user += timedelta(days=1)

        duration_user = round(duration * self.beta)
        offset_user = round(offset * self.beta)

        # TODO: Uncomment only for debugging
        #         print("Current time:", current_date)
        #         print("Task: \n\t- time:", time, "\n\t- duration:", duration, "\n\t- offset:", offset)
        #         print("Alpha:", self.alpha, end = "\n\n")
        #         print("Task converted:", task_date_user)
        #         print("Duration converted:", duration_user)
        #         print("Offset converted:", offset_user)

        return task_date_user, duration_user, offset_user

    def user_to_model(self, task_date, duration, offset,
                      current_date=datetime.now().replace(second=0, microsecond=0)):
        max_time = current_date + timedelta(minutes=self.alpha)
        task_date_model = (task_date - current_date) / (max_time - current_date)
        duration_model = timedelta(minutes=duration) / (max_time - current_date)
        offset_model = timedelta(minutes=offset) / (max_time - current_date)

        # TODO: Uncomment only for debugging
        #         print("Current time:", current_date)
        #         print("Task: \n\t- time:", task_date, "\n\t- duration:", duration, "\n\t- offset:", offset)
        #         print("Alpha:", self.alpha, end = "\n\n")
        #         print("Task converted:", task_date_model)
        #         print("Duration converted:", duration_model)
        #         print("Offset converted:", offset_model)

        return task_date_model, duration_model, offset_model

    def out_to_features(self, old_features, out, plan_date):
        old_features = old_features[0].numpy()
        out_converted = self.model_to_user(out[0], out[1], out[2])

        df = pd.DataFrame(
            columns=['Label Number',
                     'Duration',
                     'Importance',
                     'Time_Min',
                     'Date_Categorical',
                     'Date_Day',
                     'Date_Month',
                     # 'Plan_Time_Min',
                     # 'Plan_Date_Categorical',
                     # 'Plan_Date_Day',
                     # 'Plan_Date_Month'
                     ]
        )
        label_number = 0
        for i in range(4):
            if old_features[i] == 1:
                label_number = i
                break
        importance = 0
        for i in range(5, 9):
            if old_features[i] == 1:
                importance = i - 5
                break
        start_date = datetime(year=datetime.now().year, month=1, day=1) + timedelta(days=old_features[9] * 365 - 1)
        # plan_date = datetime(year=datetime.now().year, month=1, day=1) + timedelta(days=old_features[17] * 365 - 1)
        df.loc[len(df)] = {'Label Number': label_number,
                           'Duration': out_converted[1],
                           'Importance': importance,
                           'Time_Min': out_converted[0],
                           'Date_Categorical': old_features[9] * 365,
                           'Date_Day': start_date.day,
                           'Date_Month': start_date.month,
                           # 'Plan_Time_Min': old_features[14] * 1440,
                           # 'Plan_Date_Categorical': old_features[17] * 365,
                           # 'Plan_Date_Day': plan_date.day,
                           # 'Plan_Date_Month': plan_date.month
                           }

        preprocessor = Preprocessor()
        new_features = preprocessor.preprocess_activity(df, start_date, plan_date)
        return torch.tensor(np.array([new_features]), dtype=torch.float32)


# TODO: Uncomment only for debugging
# if __name__ == '__main__':
#     convertor = Converter(alpha=1440)
#     convertor.model_to_user(time=0.5, duration=0.5, offset=0.5)
#     convertor.user_to_model(task_date=datetime(2020, 12, 12, 12, 12), duration=60, offset=30)
#     convertor.out_to_features([1, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
