from tg_bot.domain.domain import Task
from Data_Test.Preprocessor import Preprocessor
import numpy as np
import datetime
import pandas as pd

class Task_generator:
    def __init__(self, day_amount, num_labels, max_importance):

        self.preprocessor = Preprocessor(alpha=1440, num_lables=num_labels, beta=300, max_importance=max_importance)

        self.day_amount = day_amount
        self.num_labels = num_labels
        self.max_importance = max_importance
        self.mean_tasks_per_day = 3

        self.X_by_days = []
        self.X_by_tasks = []

        self.Y_by_days = []
        self.Y_by_tasks = []

        self.as_features = []


    def gen_sequence(self, by_days=True, preprocess=True):
        curr_date = datetime.datetime.now().date()
        for i in range(self.day_amount):
            day_tasks = []
            day_Y = []

            # Randomly get the number of tasks for the day
            num_tasks = int(np.abs(np.random.normal(self.mean_tasks_per_day, 2)))

            # Generate num_tasks random times
            random_times = np.random.randint(0, 24 * 60, num_tasks)
            random_times = np.sort(random_times)

            # Convert to datetime.time
            random_times = [datetime.time(hour=int(time / 60), minute=time % 60) for time in random_times]

            for curr_time in random_times:
                # Get a random start time
                start_time = datetime.time(hour=np.random.randint(0, 24), minute=np.random.randint(0, 60))

                task = self.__gen_task(start_time, curr_time, curr_date)
                Y = np.array([start_time, task.duration, 0])

                if preprocess:
                    start_time = self.preprocessor._convert_time_to_alpha(start_time)
                    duration = task.duration / self.preprocessor.beta
                    Y = np.array([start_time, duration, 0], dtype=np.float32)

                    # Randomly generate label for the task
                    label = np.random.randint(0, self.num_labels)
                    task = self.preprocessor.preprocess_data(task_type="resched", data=task, label=label)


                day_tasks.append(task)
                day_Y.append(Y)
                self.X_by_tasks.append(task)
                self.Y_by_tasks.append(Y)

            self.X_by_days.append(day_tasks)
            self.Y_by_days.append(day_Y)
            curr_date += datetime.timedelta(days=1)

        if by_days:
            return self.X_by_days, self.Y_by_days

        else:
            return self.X_by_tasks, self.Y_by_tasks


    def __gen_task(self, start_time, curr_time, curr_date):
        # Get a random gamma distributed duration from 10 to 300 minutes
        duration = np.random.gamma(2.7, 30)
        duration = int(duration) if duration > 10 else 10
        duration = int(duration) if duration < 300 else 300

        # Importance - random from 0 to max_importance
        importance = np.random.randint(0, self.max_importance)

        task = Task(
            telegram_id=0,
            task_name='Task ',
            duration=duration,
            importance=importance,
            start_time=start_time,
            curr_time=curr_time,
            curr_date=curr_date,
        )
        return task

if __name__ == '__main__':
    tg = Task_generator(day_amount=2000, num_labels=4, max_importance=5)

    X, Y = tg.gen_sequence(by_days=False, preprocess=True)

    X_df = pd.DataFrame(X)
    Y_df = pd.DataFrame(Y)
    X_df.to_csv('X.csv')
    Y_df.to_csv('Y.csv')

