
import os
from ..Monitor.UserMonitor import UserMonitor

from ..db.MongoDatabase import *
import re

from ..util.data_preparation import *
from ..util.mongo_util import *
"""
path_folder_brain_flow = os.path.join("chatbots", "brain_flow")
mongoDatabase = MongoDatabase("chatbot_test", "mongodb://%s:%s@zeus.ridetel.com:27018" % ("mongoadmin", "B0t.2021"))
responses = getChatbotsResponses(mongoDatabase, 1)
"""
from ..Monitor.GeneralMonitor import GeneralMonitor
from ..Conversation import *

def test_monitor(mongoDatabase:MongoDatabase, user:str, brain_name:str)->None:


    Config.TIME_SECONDS_WAIT_FOR_FREE_MEMORY = 10
    Config.TIME_SECONDS_WAIT_FOR_USER_MONITOR_FREE_MEMORY = 10
    Config.TIME_SECONDS_WAIT_BASE_THREAD = 5
    Config.TIME_MINUTES_BOT_WAIT_TO_RESET = 1

    try:
        GeneralMonitor.init()

        type = "text"
        message = "hello"
        mobile = "+5939******041"
        mobile2 = "+593911111042"

        others = dict()
        # others["updater"] = updater
        # ensure brain is passed by reference
        GeneralMonitor.addMessageToUserMonitor(user, brain_name, mobile, message, others, mongoDatabase, type)
        time.sleep(5)
        GeneralMonitor.addMessageToUserMonitor(user, brain_name, mobile, "Message sorpresa", others, mongoDatabase, type)
        GeneralMonitor.addMessageToUserMonitor(user, brain_name, mobile2, "Message otro dos", others, mongoDatabase, type)
        time.sleep(5)
        GeneralMonitor.addMessageToUserMonitor(user, brain_name, mobile2, "Goodbye 1", others, mongoDatabase, type)
        GeneralMonitor.addMessageToUserMonitor(user, brain_name, mobile2, "Hello 2", others, mongoDatabase, type)
        
        time.sleep(5)
    finally:
        GeneralMonitor.die()
        while GeneralMonitor.existMessagesToProcess():
            print("waiting for all messages to be processed")
            time.sleep(10)

        print("Waiting for threads to finish")
        threads_to_wait = []
        for username, user_monitor_value in GeneralMonitor.users_monitors.items():
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
