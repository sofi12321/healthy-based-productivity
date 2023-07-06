import datetime

import pandas as pd
from tg_bot.domain.domain import Task, Event
from Model.sc_model import SC_LSTM
from Data.converter import Converter
from Data.Preprocessor import Preprocessor
from torch import load


class Planner:
    def __init__(self):
        # TODO: All the parameters should be in configured after training
        self.scheduler = SC_LSTM(in_features=11,
                                 lstm_layers=3,
                                 hidden=124,
                                 hidden_injector=64,
                                 out_features=3,
                                 batch_size=1,
                                 pred_interval=1440)

        # Load model weights
        self.scheduler.load_state_dict(load('Model/sc_lstm_weights.pth'))

        # Set the model to evaluation mode
        self.scheduler.eval()

        self.converter = Converter(alpha=1440)
        self.preprocessor = Preprocessor()

    def label_handling(self, task_name: str) -> int:
        """
        Converts event names from natural language to a vector of category membership:
        "Daily Routine", "Passive Rest", "Physical Activity", "Work-study".
        0 if it does not belong to the class, 1 if it does.

        :param task_name: string, name of the event
        :return: vector, length 4, event group belonging
        """
        # TODO: Danila
        label = [0, 0, 0, 0]
        return max(label)

    def preprocess_event(self, event: Event, label: int):
        """
        Processes the data about the task into the machine-understandable format.
        :param label: vector of category
        :param event: object of class Task
        :return:
            - vector of input features describing given event,
            - type of the event (non-resched),
            - vector of output features describing given event,
            - time when the event was planned by user
        """
        input_vector = self.preprocessor.preprocess_event(event, label)
        activity_type = "non-resched"
        output_vector = self.converter.user_to_model(
            task_date=datetime.datetime(year=event.date.year,
                                        month=event.date.month,
                                        day=event.date.day,
                                        hour=event.start_time.hour,
                                        minute=event.start_time.minute),
            duration=event.duration,
            offset=0
        )
        plan_time = datetime.datetime.now()

        return input_vector, activity_type, output_vector, plan_time

    def preprocess_task(self, task: Task, label: int):
        """
        Processes the data about the event into the machine-understandable format.

        :param label: vector of category
        :param task: object of class Task
        :return:
            - vector of input features describing given task,
            - type of the task (resched),
            - vector of output features describing given task (None for resched),
            - time when the task was planned by user
        """
        input_vector = self.preprocessor.preprocess_task(task, label)
        activity_type = "resched"
        output_vector = None
        plan_time = datetime.datetime.now()

        return input_vector, activity_type, output_vector, plan_time

    def call_model(self, task_type, input_features, available_time_slots, user_h, user_c):
        """
        Perform scheduling for an event or event.
        :param task_type: event for non-reschedulable, task for reschedulable
        :param input_features: vector of preprocessed features of the event
        :param available_time_slots: vector of available (free) time slots
        :param user_h: hidden state of the user
        :param user_c: cell state of the user
        :return: (,3) vector prediction of the model, and the new user states (h, c)
        """

        # Set the model states to user states
        self.scheduler.set_states(user_h, user_c)

        # Make a model prediction
        pred = self.scheduler.forward(input_features,
                                      task_type=task_type,
                                      free_time_slots=available_time_slots,
                                      save_states=True)

        # Get new user states
        new_h, new_c = self.scheduler.get_states()

        return pred, (new_h, new_c)

    def convert_output_to_schedule(self, model_output) -> [datetime.date, datetime.time, int]:
        """
        Reformat data from vector relative data into date, start time and offset.
        :param model_output: TODO: vector of 2 or 3 numbers
        :return: scheduling parameters of the event in a specified format
        """
        # TODO: Danila
        return [None, None, None]

    def fill_schedule(self, task_id: int, output: [datetime.date, datetime.time, int]):
        """
        Returns the output of the scheduler in a dictionary format, usable with database.

        :param task_id: id of the event in a database
        :param output: scheduling parameters of the event
        :return: dictionary to save data in database
        """
        # TODO: Danila
        # TODO: Deside 2 or 3 output features
        return {'task_id': task_id,
                'predicted_date': output[0],
                'predicted_start_time': output[1],
                'predicted_offset': output[2]}

    def sort_tasks(self, tasks):
        """
        Sort a list of tasks. The most important and early ones are the first.
        :param tasks: list of objects of class Task
        :return: sorted list of this tasks
        """
        tasks_0 = [task for task in tasks if task.importance == 0]
        tasks_1 = [task for task in tasks if task.importance == 1]
        tasks_2 = [task for task in tasks if task.importance == 2]
        tasks_3 = [task for task in tasks if task.importance == 3]

        tasks_0.sort(key=lambda x: (x.date, x.start_time))
        tasks_1.sort(key=lambda x: (x.date, x.start_time))
        tasks_2.sort(key=lambda x: (x.date, x.start_time))
        tasks_3.sort(key=lambda x: (x.date, x.start_time))

        return tasks_3 + tasks_2 + tasks_1 + tasks_0

    def get_model_schedule(self, tasks, events):
        """
        Collects a scheduling data about each event to update database.

        :param tasks: list of objects of class Task that were not planned before
        :param events: list of objects of class Event that were not planned before
        :return: list of dictionaries
        """
        # tasks = [Task, Task, Task, ...]
        # events = [Event, Event, Event, ...]
        resulted_schedule = []

        for event in events:
            label = self.label_handling(event.event_name)
            input_features = self.preprocess_event(event, label)
            self.call_model("event", input_features)  # TODO: Ilnur pass available_time_slots, user_h, user_c parameters

        tasks = self.sort_tasks(tasks)

        for task in tasks:
            label = self.label_handling(task.task_name)
            input_features = self.preprocess_task(task, label)
            model_output = self.call_model("event",
                                           input_features)  # TODO: Ilnur pass available_time_slots, user_h, user_c parameters
            result = self.convert_output_to_schedule(model_output)
            resulted_schedule.append(self.fill_schedule(task.task_id, result))
        return resulted_schedule

    def convert_history_to_output(self, real_date: datetime.date, real_start: datetime.time, real_duration: int,
                                  planned_duration: int) -> [float, float]:
        """
        Reformat user feedback about task completion to vector relative data (start time and offset).
        If duration == 0, then the task was not completed.

        :param real_date: date when the user performed the task
        :param real_start: time when the user started to perform the task
        :param real_duration: time during which the user performed the task
        :param planned_duration: time planned for solving the task
        :return: 2 relative parameters
        """
        # TODO: Danila
        # TODO: 2 or 3 parameters
        return [0.5, 0.5]

    def train_model(self, true_labels):
        """
        Continue the training of the model based on the user feedback.
        :param true_labels: real parameters
        """
        # TODO: Leon
        pass

    def mark_task_history(self, task: Task):
        """
        Receives a task that was completed by the user, reformat it and continue a model training on user data.
        :param task: object of class Task
        :return: True when all parts are done
        """
        true_labels = self.convert_history_to_output(task.real_date, task.real_start, task.real_duration, task.duration)
        self.train_model(true_labels)
        return True

    def print_schedule(self, tasks, events):
        """
        Prints the schedule for a day.

        :param tasks: list of objects of class Task in a schedule
        :param events: list of objects of class Event in a schedule
        :return: string with the schedule
        """
        # tasks = [Task, Task, Task, ...]
        # events = [Event, Event, Event, ...]
        return """Your schedule for today"""


if __name__ == '__main__':
    planner = Planner()
    print(planner.get_model_schedule([], []))
