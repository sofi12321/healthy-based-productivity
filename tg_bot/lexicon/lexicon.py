from typing import Optional
import logging

lexicon = {
    "en": {
        'user_not_found': "Hello, I don't know you. Please, write /start to register.",

        "first_hello": """Hello, I'm InnoPlan bot. 👋 \n\
			I'll help you to generate optimal daily schedule. 📅\n\
			I will collect a list of the tasks which you will have during the day, analyze these tasks \
			and your task history and suggest a schedule for the day which will fit your habits. 📋\n\
			My main job is to determine the order of tasks, start time, and duration. \
			All of this will be regulated using breaks between tasks. 🕐\n\n\
			How should I call you?""",
        "retry_name": "Can you, please, shorten your name a bit (at max 255 symbols)?",

        "basic_info": """Nice to meet you! 😊\n\n\
			This is the format of a particular type of input data should look like:\n\n\
			🕐 Time - '13:00' or '1:00AM' or '1:00PM'\n\
			📅 Date - 1/1/1970 (month/day/year)\n\
			⚠️ Importance - number from 0 to 3\n\
			📓 Task name - word or phrase in English. Note that currently the task is classified by its name into four categories: \n\
						      \t💪 Physical activity\n\
						      \t💻 Work and study\n\
						      \t🎨 Passive hobby\n\
						      \t🏡 Daily routine\n\
			❌ In case of absence of any task data write 'no' \n\n\

			If you forget the format of any input data you can always recall it with the command /format.""",
			
	"formats": """This is the format of a particular type of input data should look like:\n\n\
			🕐 Time - '13:00' or '1:00AM' or '1:00PM'\n\
			📅 Date - 1/1/1970 (month/day/year), today, tomorrow\n\
			⚠️ Importance - number from 0 to 3\n\
			📓 Task name - word or phrase in English. Note that currently the task is classified by its name into four categories: \n\
						      \t💪 Physical activity\n\
						      \t💻 Work and study\n\
						      \t🎨 Passive hobby\n\
						      \t🏡 Daily routine\n\
			❌ In case of absence of any task data write 'no' \n\n""",
			
	"basic_info_start_of_day": """Please write the time at which you usually start to do tasks""",

        "basic_info_end_of_day": """And at which time do you end your day?""",
        
        "end_first_hello": """Now we can start planning! Let me explain how to work with me 🤝 \n\
                        Here are 2 concepts: events and tasks:\n
		                \tEvents are ones that have a precise start time and \
		                duration, I will never change these parameters. \n\
		                \tTasks are items you want to complete during the day. Here you can specify the importance, \
		                the assumed start time and duration. Based on your history, I will modify the event parameters using time to finish the task 🏖 """,
        # "formats": """Sometimes I may not understand what you are writing. Don't worry, we have a way to communicate! When you answer my questions, please use these formats:
        # \nDate
        # \nTime""",

        "hello": "Hello, how can I help you?",
        "write_success": "Successfully saved!",
        "delete_success": "Successfully deleted!",

        "add_task": """I'm happy that you want to add a task 😊 What is the name of the task?""",
        "get_date_task": "📅 What data do you want to schedule it?",
        "get_start_time_task": """⏰ Now, what time do you want to start?""",
        "get_duration_task": """⏳ How long are you going to do it (in minutes)?""",
        "get_importance_task": "📈 Last question. How important is it (from 0 till 3)?",

        "add_event": """New event? Cool! What is its name?""",

        "event_date": "📅 What data will it be?",
        "event_start_time": "⏰ What time will it start?",
        "event_duration": f"⏳ And how long is it? (in minutes)",

        "event_want_repeat": "🔁 Is it a repeatable event?",
        "event_repeat_each_number": "How often will it repeat? For every 2nd Saturday 4 times - write 2",
        "event_repeat_each_argument": "When will it repeat? For every 2nd Saturday 4 times - write Saturday. "
                                      "Also, it may be day, week, month",
        "event_repeat_number_of_repetitions": "How many times will it happen? For every 2nd Saturday 4 times - write 4",

        "mark_history": "To mark the task completion, choose it from the list:",
        
        "marking_history_is_done": """Did you complete the task? (yes/no)""",
        "marking_history_start_time": """My congratulations 🎉 \n⏰ What time did you start task completion?""",
        "marking_history_duration": """⌛ How long did this process take? (in minutes)""",
        
        # Today is the most important day of your life. And I'm happy that you're planning it! 
        "plan": """I'll do my best to create an optimal schedule for you""",

        "list_date": "I'll show a list of tasks, but for which day? 📅",
        "list_task_event": "That's what I know:",

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
