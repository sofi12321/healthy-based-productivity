import numpy as np
import pandas as pd

from torch import load
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import re
import gdown
import datetime

from Model.sc_model import SC_LSTM
from official.nlp import optimization
from Data.converter import Converter
from Data.Preprocessor import Preprocessor
from tg_bot.domain.domain import Task, Event

ALPHA = 1440


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

        # Load NLP model
        self.nlp_model = self.nlp_load_model()

        # Set preprocessing objects
        self.converter = Converter(alpha=ALPHA)
        self.preprocessor = Preprocessor()

    def nlp_load_model(self):
        """
        Loading BERT-based task names classifier pretrained on the generated dataset

        :return: model
        """
        # Loading BERT-based task names classifier
        url = 'https://drive.google.com/u/0/uc?id=1DR6YoPst1GflO85sU2dJ9ZZV3Qi2U4vz&export=download'
        output = './model.json'
        gdown.download(url, output, quiet=False)

        # Loading weights of the classifier
        url = 'https://drive.google.com/u/0/uc?id=1kJSDhD--EFLs8jiuBUOpVU66m2-gUf7V&export=download'
        output = './model.h5'
        gdown.download(url, output, quiet=False)

        # Build model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(
            loaded_model_json,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        loaded_model.load_weights("model.h5")

        # Compile model
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(init_lr=3e-5,
                                                  num_train_steps=210,
                                                  num_warmup_steps=21,
                                                  optimizer_type='adamw')
        loaded_model.compile(optimizer=optimizer, loss=loss,
                             metrics=metrics)

        return loaded_model

    def label_handling(self, task_name: str) -> int:
        """
        Converts event names from natural language to a number of category:
        0:"Daily Routine",
        1:"Passive Rest",
        2:"Physical Activity",
        3:"Work-study".
        In case of list of task names in input, labels are generated for each task name.

        :param task_name: string, name of the events or list of events
        :return: number of group belonging
        """
        words = [task_name]
        y_pred = self.nlp_model.predict(words)
        label = np.argmax(y_pred, axis=1)
        return label[0]

    def set_available_time_slots(self, tasks, events):
        # TODO: Yaroslav
        available_time_slots = []
        return available_time_slots

    def update_available_time_slots(self, task_event, available_time_slots):
        # TODO: Yaroslav
        return available_time_slots

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

        return input_vector, activity_type, output_vector

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

        return input_vector, activity_type, output_vector

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
        prediction = self.scheduler.forward(input_features,
                                            task_type=task_type,
                                            free_time_slots=available_time_slots,
                                            save_states=True)

        # Get new user states
        new_h, new_c = self.scheduler.get_states()

        return prediction, new_h, new_c

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
                'predicted_durtion': output[2],
                'predicted_offset': output[3]}

    def get_model_schedule(self, tasks, events, user_h, user_c):
        """
        Collects a scheduling data about each event to update database.

        :param tasks: list of objects of class Task that were not planned before
        :param events: list of objects of class Event that were not planned before
        :return: list of dictionaries
        """
        # tasks = [Task, Task, Task, ...]
        # events = [Event, Event, Event, ...]
        resulted_schedule = []
        plan_time = datetime.datetime.now()
        available_time_slots = self.set_available_time_slots(tasks, events)

        for event in events:
            label = self.label_handling(event.event_name)
            input_vector, activity_type, output_vector = self.preprocess_event(event, label)
            _, user_h, user_c = self.call_model(input_vector, activity_type, available_time_slots,
                                                user_h, user_c)
            available_time_slots = self.update_available_time_slots(event, available_time_slots)

        tasks = self.sort_tasks(tasks)

        for task in tasks:
            label = self.label_handling(task.task_name)
            input_vector, activity_type, _ = self.preprocess_task(task, label)
            model_output, user_h, user_c = self.call_model(activity_type, input_vector, available_time_slots,
                                                           user_h, user_c)
            result = self.convert_output_to_schedule(model_output)
            resulted_schedule.append(self.fill_schedule(task.task_id, result))
            available_time_slots = self.update_available_time_slots(task, available_time_slots)
        return resulted_schedule, user_h, user_c

    def convert_history_to_output(self, real_date: datetime.date, real_start_time: datetime.time, real_duration: int,
                                  planned_duration: int):
        """
        Reformat user feedback about task completion to vector relative data (start time and offset).
        If duration == 0, then the task was not completed.

        :param real_date: date when the user performed the task
        :param real_start_time: time when the user started to perform the task
        :param real_duration: time during which the user performed the task
        :param planned_duration: time planned for solving the task
        :return: output vector
        """
        if real_duration == 0:
            output_vector = self.converter.user_to_model(
                task_date=datetime.datetime(year=real_date.year,
                                            month=real_date.month,
                                            day=real_date.day,
                                            hour=real_start_time.hour,
                                            minute=real_start_time.minute),
                duration=0,
                offset=0
            )
        else:
            output_vector = self.converter.user_to_model(
                task_date=datetime.datetime(year=real_date.year,
                                            month=real_date.month,
                                            day=real_date.day,
                                            hour=real_start_time.hour,
                                            minute=real_start_time.minute),
                duration=planned_duration,
                offset=real_duration - planned_duration
            )
        return output_vector

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
        # "Daily Routine", "Passive Rest", "Physical Activity", "Work-study"
        smiles = ['🏡', '🎨', '💪', '✍', '📌']
        output_schedule = "Your schedule:\n"
        base = """SMILE NAME\n🕐 START - END"""
        additional = "\n🏖 ADD_MIN min to finalize the task"
        final = "\n\n"

        order = {}
        for task in tasks:
            order[task.predicted_start_time] = ['task', task]
        for event in events:
            order[event.start_time] = ['event', event]

        sorted_order = sorted(order.keys())

        for t in sorted_order:
            output_task = ""
            if order[t][0] == 'event':
                output_task = re.sub("SMILE", smiles[-1], base)
                output_task = re.sub("NAME", order[t][1].event_name, output_task)
                output_task = re.sub("START", order[t][1].start_time.strftime("%H:%M"), output_task)
                output_task = re.sub("END",
                                     (datetime.datetime.combine(order[t][1].date, order[t][1].start_time) +
                                      datetime.timedelta(minutes=order[t][1].duration)).strftime("%H:%M"), output_task)

            elif order[t][0] == 'task':
                output_task = re.sub("SMILE", smiles[self.label_handling(order[t][1].task_name)], base)
                output_task = re.sub("NAME", order[t][1].task_name, output_task)
                output_task = re.sub("START", order[t][1].predicted_start_time.strftime("%H:%M"), output_task)
                output_task = re.sub("END", (datetime.datetime.combine(order[t][1].predicted_date,
                                                                       order[t][1].predicted_start_time)
                                             + datetime.timedelta(
                            minutes=order[t][1].predicted_duration)).strftime("%H:%M"), output_task)

                if order[t][1].predicted_offset > 5:
                    output_task += re.sub("ADD_MIN", order[t][1].predicted_offset, additional)
                output_task += final
            output_schedule += output_task
        return output_schedule


if __name__ == '__main__':
    planner = Planner()
    print(planner.get_model_schedule([], [], [], []))
