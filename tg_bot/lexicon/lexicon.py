from typing import Optional
import logging

lexicon = {
    "en": {
        'user_not_found': "Hello, I don't know you. Please, write /start to register.",

        "first_hello": """Hello, I'm a bot that will help\
                        you create a daily schedule. My name is InnoPlanner, and \
                        how should I call you?""",
        "retry_name": "Can you, please, shorten your name a bit (at max 255 symbols)?",

        "basic_info_start_of_day": """Now we now the names of each other! Let me say few words about me. \
                        You probably know that it can be hard to set a realistic time for a event during scheduling. \
                        I will analyze your event history and make your daily schedule fit your habits. \
                        Of course, I can't force you to complete tasks as I suggested, but I hope that my advice \
                        will help your productivity and life balance.\n\
                        My main job is to determine the order of tasks, start time, and duration. \
                        However, all of this will be regulated using breaks between tasks.\n\n\
                        \
                        Not to distract you, please enter the time at which you usually do tasks.\n\
                        Now write the time when you start your day in a format:\n\
                        '13:00' or '1:00AM'""",

        "basic_info_end_of_day": """And at which time do you end your day?""",
        "end_first_hello": """Now we can start planning! Let me explain how to work with me. \n\
                        Here are 2 concepts: events and tasks. Events are ones that have a precise start time and \
                        duration, I will never change these parameters. \n\
                        Tasks are items you want to complete during the day. Here you can specify the importance, \
                        the assumed start time and duration. Based on your history, I will modify the event parameters \ 
                        so that you get it done and feel good! """,
        # "formats": """Sometimes I may not understand what you are writing. Don't worry, we have a way to communicate! When you answer my questions, please use these formats:
        # \nDate
        # \nTime""",

        "hello": "Hello, how can I help you?",
        "write_success": "Successfully saved!",

        "add_task": """I'm happy that you want to add a task! What is the name of the task?""",
        "get_date_task": "When do you want to schedule it? I'm about date like 1/1/1970. If you don't care, write 'no'",
        "get_start_time_task": """Now, what time do you want to start? For example, 13:00, 1:00PM or 'no'""",
        "get_duration_task": """How long are you going to do it (in minutes)?""",
        "get_importance_task": "That's an interesting task. How important is it (from 0 till 3)?",

        "add_event": """New event? Cool! What is its name?""",

        "event_date": "When will it be? For example, 1/1/1970",
        "event_start_time": "What time will it start?",
        "event_duration": f"And how long is it? (in minutes)",

        "event_want_repeat": "Is it a repeatable event?",
        "event_repeat_each_number": "How often will it repeat? For every 2nd Saturday 4 times - write 2",
        "event_repeat_each_argument": "When will it repeat? For every 2nd Saturday 4 times - write Saturday. "
                                      "Also, it may be day, week, month",
        "event_repeat_number_of_repetitions": "How many times will it happen? For every 2nd Saturday 4 times - write 4",

        "mark_history": "I'm glad you finished the task! Choose it from the list:",
        
        "marking_history_is_done": """Did you perform the task? (yes/no)""",
        "marking_history_start_time": """When did you start task completion? ('13:00' or '1:00AM')""",
        "marking_history_duration": """How long did this process take? (in minutes)""",
       
        "plan": """Today is the most important day of your life. And I'm happy that you're planning it! I'll do my best to create an optimal schedule for you""",

        "list_date": "Please, enter date to list",
        "list_task_event": "Tasks and event of the given date are listed below",

        # retries
        "retry_int": "Please, write an integer",
        "retry_optional_time": "Small reminder, I understand time only in this formats:\n13:00 or 1:00PM. If it does not have a specific start time, you can write 'no'",
        "retry_optional_date": """Sorry, I didn't understand you. Please, write date in the following format:\n
        dd/mm/yyyy. For example, 1/1/1970""",
        "retry_time": """Ups, can you, please, repeat the time in this format: '13:00' or '1:00AM'?""",
        "retry_trouble": "Sorry, some mistake have occured. Please, try again",
        "error": "I got unexpected error: \n{}",
        "retry_yesno": "This is a simple question, just write 'yes' or 'no'",

        "retry_date": "Please, write date in format: 1/1/1970"

        
        # "ask_history": """Please, enter history in the following format:\n\t
        # [start_time]-[end_time]-[is_done (true or false)]""",
        # "retry_event": """Please, enter event in the following format:\n\t
        # [event_name]-[start_time]-[duration (in minutes)]-[Number] [day | {day of week (e.g. Sunday)} | month] [Number of events]""",
        # "retry_history": """Please, enter history in the following format:\n\t
        # [start_time]-[end_time]-[is_done (true or false)]""",
        
    }
}


def get_lexicon_with_argument(message_args: str, *args) -> Optional[str]:
    if not isinstance(message_args, str):
        logging.error("Got unexpected type in lexicon")
        return ""

    if message_args not in lexicon["en"].keys():
        logging.error("Got unexpected argument in lexicon: %s", message_args)
        return ""

    try:
        return lexicon["en"][message_args].format(*args)
    except IndexError:
        return ""
