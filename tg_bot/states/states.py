from aiogram.dispatcher.filters.state import State, StatesGroup


class GetBasicInfo(StatesGroup):
    StartEndOfDay = State()


class BaseState(StatesGroup):
    Unit = State()


class GetTaskInfo(StatesGroup):
    TaskNameState = State()
    TaskDurationState = State()
    TaskImportanceState = State()
    TaskComplexityState = State()
    TaskStartTimeState = State()
    TaskDateState = State()


class GetEventInfo(StatesGroup):
    SpecificInput = State()


class MarkHistory(StatesGroup):
    ChooseTask = State()
    MarkingHistory = State()


UnitState = State()