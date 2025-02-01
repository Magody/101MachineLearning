import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from tensorflow.keras import models
from tensorflow.python.util.dispatch import dispatch

from models.Monitor.GeneralMonitor import GeneralMonitor
from models.Monitor.UserMonitor import UserMonitor

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

from models.ChatbotRetrieval import ChatbotRetrieval
import random
import json
import pickle
import numpy as np

from tensorflow.keras.models import load_model

from models.util.nltk_transforms import *
from models.BrainRetrieval import BrainRetrieval

from models.util.data_preparation import *
import sys

from models.Conversation import *
from models.db.MongoDatabase import *
import os

from thread_messengers import thread_messengers

# the chatbot telegram set the brain with program execution

# console configuration
global brain_name
arg_list = sys.argv
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(arg_list))

Config.setEnvironment(os.path.join(".env.json"))
# python messenger_bot_telegram.py test brain_telegram "1659508668:AAEv7r7qLb0zUS8pl_B4dvvLq8QJIzmaOW0"
mode = arg_list[1]
brain_name = arg_list[2]
token = arg_list[3] # 1659508668:AAEv7r7qLb0zUS8pl_B4dvvLq8QJIzmaOW0 ->>> TODO: SAVE THIS IN DATABASE


def echo(update: Update, telegramContext: CallbackContext) -> None:
    
    # messenger configuration
    global brain_name

    mongoDatabase = MongoDatabase(
        Config.DB_NAME,
        f"mongodb://{Config.MONGODB_USER}:{Config.MONGODB_PASSWORD}@{Config.HOST}:{Config.MONGO_DB_PORT}/{Config.DB_NAME}?retryWrites=true&w=majority",
    )   

    others = dict()
    others["updater"] = telegramContext
    others["device_id"] = ""
    others["chat_w_id"] = update.message.chat_id
    
    unique_id = str(update.message.chat_id)
    mobile = unique_id
    message: str = update.message.text

    type = "text"


    if message.lower() == "reset":
        GeneralMonitor.resetContextOfUserBrainConversation(Config.USERNAME, mongoDatabase, "telegram", brain_name, mobile)

        update.message.reply_text(f"Si la conversación no estaba en memoria entonces el contexto para la conversación id={unique_id} se reseteó, caso contrario inténtelo luego de unos minutos si es fase de test")
        return

    ###########
    # Generic configuration
    GeneralMonitor.addMessageToUserMonitor(Config.USERNAME, brain_name, mobile, message, others, mongoDatabase, type)
          


def main() -> None:

    """Start the bot."""

    # Create the Updater and pass it your bot's token.
    updater = Updater(token) # 1659508668:AAEv7r7qLb0zUS8pl_B4dvvLq8QJIzmaOW0

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    # dispatcher.add_handler(CommandHandler("random", random))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()




if __name__ == '__main__':
    try:
        GeneralMonitor.init(thread_messengers, mode)
        main()
    except Exception as error:
        print("Error", error)
    finally:
        GeneralMonitor.die()
        while GeneralMonitor.existMessagesToProcess():
            print("waiting for all messages to be processed")
            time.sleep(10)

        print("Waiting for threads to finish")
        threads_to_wait = []
        for Config.USERNAME, user_monitor_value in GeneralMonitor.users_monitors.items():
            userMonitor: UserMonitor = user_monitor_value["user_monitor"]
            for brain_name, brain_value in userMonitor.brain_conversations.items():

                for unique_id, conv in brain_value["conversations"]:
                    conversation: Conversation = conv
                    threads_to_wait.append(conversation.thread)


        for thread in threads_to_wait:
            if thread:
                print("waiting for", unique_id)
                thread.join()
        print("End of program")