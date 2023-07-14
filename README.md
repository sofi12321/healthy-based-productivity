# **InnoPlan bot**

According to research of hh.ru and AIBY*, every second respondent felt burned out. The main reasons were emotional exhaustion and a busy schedule, lack of time for work and personal tasks. The reason could be an “idealized” planning - when a person is too optimistic in estimating the task duration, taking no breaks or overloading the schedule.

<p align="center">
    <img width="200" src="/static/logo.png" alt="InnoPlan logo">
</p>

The solution is the **InnoPlan bot** by Healthy Based Productivity startup. It's a chatbot in Telegram that builds a realistic schedule of user tasks. Unlike other autoschedulers, our solution is *personalized* because we build a plan based on a person's *activity history*.

This solution is not yet on the market, be one of the first who will try it! [Link to our chatbot @InnoPlan_bot](https://t.me/Innoplan_bot).

## Table of Contents

- Getting Started
- Features
- Setup Instructions
- Usage
- Contributing

## Getting Started

InnoPlan bot is created to make the schedule similar to the daily workflow of a person. It uses the history of completed tasks to predict the best schedule for today. 
Our product offers the opportunity to optimize a person's schedule based on his or her activity history.
Let's consider an example where the user systematically estimates his or her capabilities too optimistically and allocates less time to tasks than he or she actually spends on them. In this case, the system notes this issue and generates a more objective schedule for the day. The ability to create a schedule based on activity history is an innovation of our project.
The project implementation is a chatbot in Telegram. In it, users will be able to work with the list of tasks and events - add, delete and mark their completion and receive the generated day schedule.

Our project offers many benefits to users, such as
* avoiding mistakes made by people in the planning process;
* saving time, as there is no need to analyze and schedule schedules yourself;
* improved wellness due to the absence of worries about failed plans.

## Features
1. Personalization - the system predicts the best time parameters for the task based on user activity history.
2. Tasks names may be written in Natural languge. The system classify them into 4 categories: Sport, Passive Rest, Daily Routine, Study&Work.
3. All tasks are stored in the database, so users can plan their day later.
4. User 

## Setup Instructions

To set up the Healthy Productivity Bot locally, please follow the instructions below:

1. Clone this repository to your local machine using the command:
```
git clone https://github.com/sofi12321/healthy-based-productivity
```

2. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

3. Obtain the necessary API keys and tokens:

   - Create a new bot on Telegram and obtain the API token.
   - If required, acquire any additional credentials for external APIs or services used by the bot.

4. Configure the bot:

   - Replace the API_TOKEN variable in the code with your obtained Telegram API token.
   - Set up any other required credentials or environment variables related to external services.

5. Run the bot:

python tgbot/main.py


Congratulations! The Healthy Productivity Bot is now up and running on your machine.

## Usage

To use the Healthy Productivity Bot, follow these steps:

1. Open the Telegram app and search for the bot using the username you provided during setup.

2. Start a conversation with the bot by sending the /start command.

3. Interact with the bot using the available commands and options to receive task suggestions, create schedules, and more.

## Contributing

Contributions are always welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork this repository.

2. Create a new branch in your forked repository for your changes.

3. Make your desired changes and improvements.

4. Commit and push your changes to your branch.

5. Open a pull request, detailing the changes you've made and explaining why they should be merged.

6. Wait for a maintainer to review your pull request. Once approved, your changes will be merged into the main repository.

Thank you for your valuable contributions!
