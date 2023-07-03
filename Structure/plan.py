import datetime
from tg_bot.domain.domain import Task, Event


def label_handling(task_name: str) -> [int, int, int, int]:
    """
    Converts event names from natural language to a vector of category membership:
    "Daily Routine", "Passive Rest", "Physical Activity", "Work-study".
    0 if it does not belong to the class, 1 if it does.

    :param task_name: string, name of the event
    :return: vector, length 4, event group belonging
    """
    # TODO: Danila
    label = [0, 0, 0, 0]
    return label


def preprocess_event(event: Event, label: [int, int, int, int]):
    """
    Processes the data about the task into the machine-understandable format.

    :param label: vector of category
    :param event: object of class Task
    :return: vector of input features describing given task
    """
    # TODO: Yaroslav
    result = label + [event.duration]
    return result


def preprocess_task(task: Task, label: [int, int, int, int]):
    """
    Processes the data about the event into the machine-understandable format.

    :param label: vector of category
    :param task: object of class Task
    :return: vector of input features describing given event
    """
    # TODO: Yaroslav
    result = label + [task.duration, task.importance]
    return result


def call_model(task_type, input_features):
    """
    Perform scheduling for a event or event.

    :param task_type: event for non-reschedulable, task for reschedulable
    :param input_features: vector of preprocessed features of the event
    :return: TODO: vector of 2 or 3 numbers
    """
    # TODO: Leon
    return [0, 0, 0]


def convert_output_to_schedule(model_output) -> [datetime.date, datetime.time, int]:
    """
    Reformat data from vector relative data into date, start time and offset.
    :param model_output: TODO: vector of 2 or 3 numbers
    :return: scheduling parameters of the event in a specified format
    """
    # TODO: Danila
    return [None, None, None]


def fill_schedule(telegram_id: int, output: [datetime.date, datetime.time, int]):
    """
    Returns the output of the scheduler in a dictionary format, usable with database.

    :param telegram_id: id of the event in a database
    :param output: scheduling parameters of the event
    :return: dictionary to save data in database
    """
    # TODO: Danila
    # TODO: Deside 2 or 3 output features
    return {'telegram_id': telegram_id,
            'predicted_date': output[0],
            'predicted_start_time': output[1],
            'predicted_offset': output[2]}


def sort_tasks(tasks):
    """
    Sort a list of tasks. The most important and early ones are the first.
    :param tasks: list of objects of class Task
    :return: sorted list of this tasks
    """
    return tasks


def get_model_schedule(tasks, events):
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
        label = label_handling(event.event_name)
        input_features = preprocess_event(event, label)
        call_model("event", input_features)

    tasks = sort_tasks(tasks)

    for task in tasks:
        label = label_handling(task.task_name)
        input_features = preprocess_task(task, label)
        model_output = call_model("event", input_features)
        result = convert_output_to_schedule(model_output)
        resulted_schedule.append(fill_schedule(task.telegram_id, result))
    return resulted_schedule


if __name__ == '__main__':
    print(get_model_schedule([], []))
