import numpy as np
import pandas as pd
import torch

from torch import load
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import re
import gdown
import datetime
import warnings

from Model.sc_model import SC_LSTM
from Data.converter import Converter
from Data.Preprocessor import Preprocessor
from tg_bot.domain.domain import Task, Event


class Planner:
    def __init__(self, alpha=1440):
        # TODO: All the parameters should be in configured after training
        self.scheduler = SC_LSTM(
            in_features=22,
            lstm_layers=1,
            hidden=124,
            hidden_injector=64,
            out_features=3,
            batch_size=1,
            pred_interval=alpha,
        )

        # Load model weights
        # TODO: UNCOMMENT THIS
        self.scheduler.load_state_dict(load("Model/sc_lstm_weights.pth"))

        # Set the model to evaluation mode
        self.scheduler.eval_model()

        # Load NLP model
        self.nlp_model = self.nlp_load_model()

        # Set preprocessing objects
        self.converter = Converter(alpha=alpha)
        self.preprocessor = Preprocessor()
        # Initially all time slots are free
        self.available_time_slots = [[0, 1]]

    def nlp_load_model(self):
        """
        Loading BERT-based task names classifier pretrained on the generated dataset

        :return: model
        """
        # # Loading BERT-based task names classifier
        # url = 'https://drive.google.com/u/0/uc?id=1DR6YoPst1GflO85sU2dJ9ZZV3Qi2U4vz&export=download'
        # output = './model.json'
        # gdown.download(url, output, quiet=False)
        #
        # # Loading weights of the classifier
        # url = 'https://drive.google.com/u/0/uc?id=1kJSDhD--EFLs8jiuBUOpVU66m2-gUf7V&export=download'
        # output = './model.h5'
        # gdown.download(url, output, quiet=False)

        # Build model
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(
            loaded_model_json, custom_objects={"KerasLayer": hub.KerasLayer}
        )
        loaded_model.load_weights("model.h5")

        # Compile model
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-5)
        loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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
        """
        Set available time slots based on the already scheduled events and tasks.
        :param tasks: list of objects Task, not all were scheduled
        :param events: list of objects Event, not all were scheduled
        """
        self.available_time_slots = [[0, 1]]
        for event in events:
            if event.was_scheduled:
                self.update_available_time_slots_event(event)
        for task in tasks:
            if task.predicted_start:
                task_output = self.convert_history_to_output(
                    task.predicted_date,
                    task.predicted_start,
                    task.predicted_duration,
                    task.predicted_duration + task.predicted_offset,
                )
                self.update_available_time_slots_task(task_output)

    def update_available_time_slots_event(self, event):
        """
        Update available time slots based on the scheduled event
        :param event: object of class Event, scheduled event
        """
        event_start, event_dur, _ = self.convert_history_to_output(
            event.date, event.start_time, event.duration, event.duration
        )
        self.available_time_slots = self.update_slot(
            event_start, event_dur, self.available_time_slots
        )

    def update_available_time_slots_task(self, prediction):
        """
        Update available time slots based on the scheduled task
        :param prediction: tensor contains 3 number alpha related: start_time, duration, offset
        """
        time, duration, offset = prediction[0], prediction[1], prediction[2]
        # print("вход слотов", time, duration, offset)
        self.available_time_slots = self.update_slot(
            time, max(duration, duration + offset), self.available_time_slots
        )

    def update_slot(self, start_time, duration, time_slots):
        """
        Updates one time slot
        :param start_time: [0,1] start time alpha-related
        :param duration: [0,1] duration alpha-related
        :param time_slots: list of available slots
        """
        # print("dgjdkghkd", start_time, duration)
        for i in range(len(time_slots)):
            # Should change start of the slot
            if start_time < 0:
                start_time = 0
            elif start_time + duration > 1:
                duration = start_time + duration - 1
            if time_slots[i][0] == start_time:
                time_slots[i][0] += duration
                if time_slots[i][0] >= time_slots[i][1]:
                    if time_slots[i][0] > time_slots[i][1]:
                        print("Something went wrong. Time slots are overlapping 1")
                    del time_slots[i]

            # Should change end of the slot
            elif time_slots[i][0] + time_slots[i][1] == start_time + duration:
                time_slots[i][1] -= duration
                if time_slots[i][0] >= time_slots[i][1]:
                    if time_slots[i][0] > time_slots[i][1]:
                        print("Something went wrong. Time slots are overlapping 2")
                    del time_slots[i]

            # Should divide slot in 2 slots, center cut
            elif (
                time_slots[i][0] < start_time < start_time + duration < time_slots[i][1]
            ):
                new_t_s = [start_time + duration, time_slots[i][1]]
                time_slots[i][1] = start_time
                time_slots.insert(i + 1, new_t_s.copy())

            # misunderstanding - time slots are overlapping
            elif (
                i + 1 < len(time_slots)
                and time_slots[i][0] < start_time < time_slots[i + 1][0]
            ):
                # gets into time slot
                print("Something went wrong. Time slots are overlapping 3")
                time_slots[i][1] = start_time
                if time_slots[i + 1][0] < start_time + duration:
                    time_slots[i + 1][0] = start_time + duration
            else:
                continue
            break
        return time_slots

    def preprocess_event(self, event: Event, label: int, plan_time):
        """
        Processes the data about the task into the machine-understandable format.
        :param plan_time: time when user call /plan
        :param label: vector of category
        :param event: object of class Task
        :return:
            - vector of input features describing given event,
            - type of the event (non-resched),
            - vector of output features describing given event,
            - time when the event was planned by user
        """
        input_vector = self.preprocessor.preprocess_event(event, label, plan_time)
        activity_type = "non-resched"

        return input_vector, activity_type

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

    def preprocess_task(self, task: Task, label: int, plan_time):
        """
        Processes the data about the event into the machine-understandable format.

        :param plan_time:
        :param label: vector of category
        :param task: object of class Task
        :return:
            - vector of input features describing given task,
            - type of the task (resched),
            - vector of output features describing given task (None for resched),
            - time when the task was planned by user
        """
        # TODO: Yaroslav, план тайм добавить в обработку
        input_vector = self.preprocessor.preprocess_task(task, label, plan_time)
        activity_type = "resched"

        return input_vector, activity_type

    def call_model(
        self, input_features, task_type, available_time_slots, user_h, user_c
    ):
        """
        Perform scheduling for an event or event.
        :param task_type: event for non-reschedulable, task for reschedulable
        :param input_features: vector of preprocessed features of the event
        :param available_time_slots: vector of available (free) time slots
        :param user_h: hidden state of the user
        :param user_c: cell state of the user
        :return: (,3) vector prediction of the model, and the new user states (h, c)
        """

        # Convert input_features to the torch tensor
        input_features = np.array(input_features).astype(np.float64)
        input_features = torch.tensor(input_features, dtype=torch.float32)
        input_features = input_features.unsqueeze(0)

        print(available_time_slots)

        # Set the model states to user states
        self.scheduler.set_states(user_h, user_c)

        with torch.no_grad():
            # Make a model prediction
            prediction = self.scheduler.forward(
                input_features,
                task_type=task_type,
                free_time_slots=available_time_slots,
                save_states=True,
            )

        # Get new user states
        new_h, new_c = self.scheduler.get_states()

        # TODO: DELETE THIS
        print(f"Output prediction: {prediction}")

        if isinstance(prediction, torch.Tensor):
            prediction = prediction[0].numpy()

        # [[1,2,3]]
        return prediction, new_h, new_c

    def convert_output_to_schedule(self, task_id: int, prediction, plan_time):
        """
        Reformat data from vector relative data into date, start time, duration and offset.
        Returns the output of the scheduler in a dictionary format, usable with database.

        :param task_id: id of the task in a database
        :param prediction: predicted scheduling parameters
        :param plan_time: datetime when /plan was called
        :return: dictionary to save data in database
        """
        print("Prediction", prediction)
        time, duration, offset = prediction[0], prediction[1], prediction[2]
        task_datetime_user, duration_user, offset_user = self.converter.model_to_user(
            time, duration, offset, current_date=plan_time
        )
        return {
            "task_id": task_id,
            "predicted_date": task_datetime_user.date(),
            "predicted_start_time": task_datetime_user.time(),
            "predicted_duration": duration_user,
            "predicted_offset": offset_user,
        }

    def get_model_schedule(
        self,
        tasks,
        events,
        user_h,
        user_c,
        plan_time=datetime.datetime.now().replace(second=0, microsecond=0),
    ):
        """
        Collects a scheduling data about each event to update database.

        :param tasks: list of objects of class Task that were not planned before
        :param events: list of objects of class Event that were not planned before
        :param user_h: user history from model
        :param user_c: user context from model
        :param plan_time: datetime when /plan was called
        :return: list of dictionaries
        """
        # tasks = [Task, Task, Task, ...]
        # events = [Event, Event, Event, ...]
        resulted_schedule = []
        self.set_available_time_slots(tasks, events)

        # Keep only those tasks that were not scheduled before
        tasks_new, events_new = [], []
        for event in events:
            if not event.was_scheduled:
                events_new.append(event)
        for task in tasks:
            if not task.predicted_start:
                tasks_new.append(task)

        # Schedule events first. They must be in their places
        for event in events_new:
            # TODO: UNCOMMENT PLUG
            label = 0
            # label = self.label_handling(event.event_name)
            input_vector, activity_type = self.preprocess_event(event, label, plan_time)
            _, user_h, user_c = self.call_model(
                input_vector, activity_type, self.available_time_slots, user_h, user_c
            )
            self.update_available_time_slots_event(event)

        # Sort tasks to help the model
        tasks_new = self.sort_tasks(tasks_new)

        for task in tasks_new:
            # TODO: UNCOMMENT PLUG
            label = 0
            # label = self.label_handling(task.task_name)
            input_vector, activity_type = self.preprocess_task(task, label, plan_time)

            # start_time duration offset
            model_output, user_h, user_c = self.call_model(
                input_vector, activity_type, self.available_time_slots, user_h, user_c
            )
            # TODO CHECK SHAPE model output !!!
            task_schedule = self.convert_output_to_schedule(
                task.task_id, model_output, plan_time
            )
            resulted_schedule.append(task_schedule)

            self.update_available_time_slots_task(model_output)

        return resulted_schedule, user_h, user_c

    def convert_history_to_output(
        self,
        real_date: datetime.date,
        real_start_time: datetime.time,
        real_duration: int,
        planned_duration: int,
    ):
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
                task_date=datetime.datetime(
                    year=real_date.year,
                    month=real_date.month,
                    day=real_date.day,
                    hour=real_start_time.hour,
                    minute=real_start_time.minute,
                ),
                duration=0,
                offset=0,
            )
        else:
            output_vector = self.converter.user_to_model(
                task_date=datetime.datetime(
                    year=real_date.year,
                    month=real_date.month,
                    day=real_date.day,
                    hour=real_start_time.hour,
                    minute=real_start_time.minute,
                ),
                duration=planned_duration,
                offset=real_duration - planned_duration,
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
        true_labels = self.convert_history_to_output(
            task.real_date, task.real_start, task.real_duration, task.duration
        )
        self.train_model(true_labels)
        return True

    def print_schedule(self, tasks, events):
        """
        Prints the schedule for a day.

        :param tasks: list of objects of class Task in a schedule
        :param events: list of objects of class Event in a schedule
        :return: string with the schedule
        """

        # "Daily Routine", "Passive Rest", "Physical Activity", "Work-study"
        smiles = ["🏡", "🎨", "💪", "✍", "📌"]
        output_schedule = "Your schedule:\n"
        base = """SMILE NAME\n🕐 START - END"""
        additional = "\n🏖 ADD_MIN min to finalize the task"
        final = "\n\n"

        tasks_not_done = []
        flag_plan = False
        # Sort all tasks and events by the start time
        order = {}
        for task in tasks:
            if task.predicted_start:
                order[task.predicted_start] = ["task", task]
            else:
                tasks_not_done.append(task)
        for event in events:
            if not event.was_scheduled:
                flag_plan = True
            order[event.start_time] = ["event", event]
        sorted_order = sorted(order.keys())

        for t in sorted_order:
            output_task = ""
            if order[t][0] == "event":
                output_task = re.sub("SMILE", smiles[-1], base)
                output_task = re.sub("NAME", order[t][1].event_name, output_task)
                output_task = re.sub(
                    "START", order[t][1].start_time.strftime("%H:%M"), output_task
                )
                output_task = re.sub(
                    "END",
                    (
                        datetime.datetime.combine(
                            order[t][1].date, order[t][1].start_time
                        )
                        + datetime.timedelta(minutes=order[t][1].duration)
                    ).strftime("%H:%M"),
                    output_task,
                )

            elif order[t][0] == "task":
                output_task = re.sub(
                    "SMILE", smiles[self.label_handling(order[t][1].task_name)], base
                )
                output_task = re.sub("NAME", order[t][1].task_name, output_task)
                output_task = re.sub(
                    "START", order[t][1].predicted_start.strftime("%H:%M"), output_task
                )
                output_task = re.sub(
                    "END",
                    (
                        datetime.datetime.combine(
                            order[t][1].predicted_date, order[t][1].predicted_start
                        )
                        + datetime.timedelta(minutes=order[t][1].predicted_duration)
                    ).strftime("%H:%M"),
                    output_task,
                )

                if order[t][1].predicted_offset >= 1:
                    output_task += re.sub(
                        "ADD_MIN", str(int(order[t][1].predicted_offset)), additional
                    )
            output_task += final
            output_schedule += output_task

        if flag_plan:
            if len(tasks_not_done) > 0:
                output_schedule += (
                    "Please, call /plan to add all tasks and events in the schedule"
                )
            else:
                output_schedule += "Please, call /plan to add not scheduled events"
        elif len(tasks_not_done) < 1:
            output_schedule += "Please, call /plan to add not scheduled tasks"
        elif len(output_schedule) < 1:
            output_schedule += (
                "Please, add tasks and events first using /add_task or /add_event"
            )
        return output_schedule


if __name__ == "__main__":
    planner = Planner()
    tasks = [
        Task(
            telegram_id=0,
            task_name="sport",
            importance=2,
            start_time=datetime.time(13, 20),
            duration=20,
            date=datetime.datetime.now().date(),
            # predicted_start=datetime.time(13, 20), predicted_duration=20, predicted_offset=5,
            predicted_date=datetime.datetime.now().date(),
        ),
        Task(
            telegram_id=1,
            task_name="music",
            importance=1,
            duration=40,
            start_time=datetime.time(17, 20),
            date=datetime.datetime.now().date(),
            # predicted_start=datetime.time(17, 20), predicted_duration=40, predicted_offset=10,
            predicted_date=datetime.datetime.now().date(),
        ),
    ]
    events = [
        Event(
            telegram_id=3,
            event_name="lesson_1",
            start_time=datetime.time(15, 0),
            duration=90,
            date=datetime.datetime.now().date(),
        ),
        Event(
            telegram_id=5,
            event_name="lesson_2",
            duration=120,
            start_time=datetime.time(20, 20),
            date=datetime.datetime.now().date(),
        ),
    ]
    planner.get_model_schedule(
        tasks,
        events,
        [[0] * 124] * 1,
        [[0] * 124] * 1,
        plan_time=datetime.datetime.now().replace(
            hour=8, minute=0, second=0, microsecond=0
        ),
    )
    print(planner.print_schedule(tasks, events))
