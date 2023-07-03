import datetime
from tg_bot.domain.domain import Task, Event


def convert_history_to_output(real_date: datetime.date, real_start: datetime.time, real_duration: int,
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


def train_model(true_labels):
    """
    Continue the training of the model based on the user feedback.
    :param true_labels: real parameters
    """
    # TODO: Leon
    pass


def mark_task_history(task: Task):
    """
    Receives a task that was completed by the user, reformat it and continue a model training on user data.
    :param task: object of class Task
    :return: True when all parts are done
    """
    true_labels = convert_history_to_output(task.real_date, task.real_start, task.real_duration, task.duration)
    train_model(true_labels)
    return True
