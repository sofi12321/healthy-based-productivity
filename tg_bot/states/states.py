from aiogram.dispatcher.filters.state import State, StatesGroup


class GetBasicInfo(StatesGroup):
    NameOfUser = State()
    StartOfDay = State()
    EndOfDay = State()


class GetTaskInfo(StatesGroup):
    TaskNameState = State()
    TaskDurationState = State()
    TaskImportanceState = State()
    TaskComplexityState = State()
    TaskStartTimeState = State()
    TaskDateState = State()


class GetEventInfo(StatesGroup):
    EventNameState = State()
    EventStartTimeState = State()
    EventDurationState = State()
    EventDateState = State()
    EventRepeatIsNeeded = State()
    EventRepeatEachNumberState = State()
    EventRepeatEachArgumentState = State()
    EventRepeatNumberOfRepetitions = State()


class MarkHistory(StatesGroup):
    ChooseTask = State()
    MarkingHistoryStartTime = State()
    MarkingHistoryDuration = State()
    MarkingHistoryIsDone = State()


class List(StatesGroup):
    ChooseDate = State()
    ListingTasksEvents = State()
    ShowingTaskEvent = State()
