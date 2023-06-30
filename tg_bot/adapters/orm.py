from sqlalchemy import (
    Column,
    MetaData,
    Integer,
    String,
    ForeignKey,
    Text,
    Enum,
    Table,
    Boolean,
    Date,
    Time,
    CheckConstraint,
    Date,
)
from sqlalchemy.orm import registry, mapper, relationship
import datetime

from domain import domain

mapper_registry = registry()
metadata = mapper_registry.metadata

users_table = Table(
    "Users",
    metadata,
    Column("telegram_id", Integer, primary_key=True, unique=True),
    Column("name", Text, nullable=False),
    Column("start_time", Time, nullable=False),
    Column("end_time", Time, nullable=False),
)

tasks_table = Table(
    "Tasks",
    metadata,
    Column("task_id", Integer, primary_key=True, unique=True, autoincrement=True),
    Column(
        "telegram_id",
        Integer,
        ForeignKey("Users.telegram_id", unupdate="CASCADE", ondelete="CASCADE"),
    ),
    Column("task_name", Text, nullable=False),
    Column("duration", Integer, nullable=False),
    Column("importance", Integer, nullable=False),
    Column("start_time", Time, nullable=True),
    Column("date", Date, default=datetime.date.today()),
    Column("is_done", Boolean, default=False),
    Column("real_start", Time, nullable=True),
    Column("real_duration", Integer, nullable=True),
    Column("real_date", Date, nullable=True),
    # Constraints
    CheckConstraints("duration > 0"),
    CheckConstraints("importance >= 0 and importance <= 3"),
    CheckConstraints("real_duration is null or real_duration > 0"),
    CheckConstraints(
        "is_done is true and real_start is not null and real_duration is not null and real_date is not null"
    ),
)

events_table = Table(
    "Events",
    metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("event_name", Text, nullable=False),
    Column("start_time", Time, nullable=False),
    Column("duration", Integer, nullable=False),
    Column("date", Date, default=datetime.date.today()),
    Column("repeat_number", Integer, default=0),
    # Constraits
    CheckConstraint("duration > 0"),
    CheckConstraint("repeat_number >= 0"),
)


def start_mappers():
    user_mapper = mapper_registry.map_imperatively(domain.BasicUserInfo, users_table)
    task_mapper = mapper_registry.map_imperatively(domain.Task, tasks_table)
    event_mapper = mapper_registry.map_imperatively(domain.Event, events_table)


def create_all(engine):
    metadata.bind(engine)
    metadata.create_all()


def delete_all(engine):
    metadata.drop_all(engine)
