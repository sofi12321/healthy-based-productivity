from domain import domain
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional, List
import datetime

__repo_users = None
__repo_events = None
__repo_tasks = None


class UsersRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_user(self, user: domain.BasicUserInfo):
        self.session.add(user)
        self.session.commit()

    def get_user_by_id(self, user_telegram_id: int) -> Optional[domain.BasicUserInfo]:
        return self.session.get(domain.BasicUserInfo, {"telegram_id": user_telegram_id})

    def get_delete_user(self, user: domain.BasicUserInfo):
        self.session.delete(user)
        self.session.commit()


class EventsRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_event(self, event: domain.Event):
        self.session.add(event)
        self.session.commit()

    def get_event_by_id(self, event_id: int, user_id: int) -> Optional[domain.Event]:
        return self.session.get(
            domain.Event, {"event_id": event_id, "telegram_id": user_id}
        )

    def get_events_in_range(self, user_id: int, delta_hours: int, from_date: datetime.date, from_time: datetime.time) -> [domain.Event]:
        delta_days = delta_hours % 24
        delta_time = delta_hours // 24

        from_datetime = datetime.datetime.combine(from_date, from_time)
        to_datetime = from_datetime + datetime.timedelta(days=delta_days, hours=delta_time)

        events = self.session.query(domain.Event).filter_by(telegram_id=user_id).filter(
            domain.Event.date.between(from_datetime.date(), to_datetime.date())
        ).all()

        return_arr = []

        for event in events:
            event_datetime = datetime.datetime.combine(event.date, event.start_time)
            event_duration = datetime.timedelta(minutes=event.duration)

            if from_datetime <= event_datetime + event_duration <= to_datetime:
                return_arr.append(event)

        return return_arr

    def get_all_events_for_specific_date(
        self, date: datetime.date, user_id: int
    ) -> List[domain.Event]:
        return (
            self.session.query(domain.Event)
            .filter_by(telegram_id=user_id, date=date)
            .all()
        )

    def delete_event(self, event: domain.Event):
        self.session.delete(event)
        self.session.commit()

    def add_many_events(self, events: [domain.Event]):
        self.session.add_all(events)
        self.session.commit()


class TasksRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_task(self, task: domain.Task):
        self.session.add(task)
        self.session.commit()

    def get_task_by_id(self, task_id: int, user_id: int) -> Optional[domain.Task]:
        return self.session.query(domain.Task).filter_by(task_id=task_id, telegram_id=user_id).one_or_none()

    def get_all_tasks_for_specific_date(
        self, date: datetime.date, user_id: int
    ) -> List[domain.Task]:
        return (
            self.session.query(domain.Task)
            .filter_by(telegram_id=user_id, date=date)
            .all()
        )

    def get_active_tasks(self, user_id: int) -> List[domain.Task]:
        return self.session.query(domain.Task).filter_by(telegram_id=user_id, is_done=False).all()

    def get_task_in_range(self, user_id: int, delta_hours: int, from_date: datetime.date, from_time: datetime.time) -> [domain.Task]:
        delta_days = delta_hours % 24
        delta_time = delta_hours // 24

        from_datetime = datetime.datetime.combine(from_date, from_time)
        to_datetime = from_datetime + datetime.timedelta(days=delta_days, hours=delta_time)

        tasks = self.session.query(domain.Task).filter_by(telegram_id=user_id).filter(
            domain.Task.date.between(from_datetime.date(), to_datetime.date())
        ).all()

        return_arr = []

        for task in tasks:
            task_datetime = datetime.datetime.combine(task.date, task.start_time)
            task_duration = datetime.timedelta(minutes=task.duration)

            if from_datetime <= task_datetime + task_duration <= to_datetime:
                return_arr.append(task)

        return return_arr

    def delete_task(self, task):
        self.session.delete(task)
        self.session.commit()

    def update(self):
        self.session.commit()


def initialize_repositories(session: Session):
    global __repo_events, __repo_tasks, __repo_users

    if not __repo_events:
        __repo_events = EventsRepository(session)

    if not __repo_tasks:
        __repo_tasks = TasksRepository(session)

    if not __repo_users:
        __repo_users = UsersRepository(session)


def get_events_repo():
    global __repo_events
    return __repo_events


def get_tasks_repo():
    global __repo_tasks
    return __repo_tasks


def get_users_repo():
    global __repo_users
    return __repo_users
