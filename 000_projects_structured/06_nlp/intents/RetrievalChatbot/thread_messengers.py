import copy
import threading
from types import TracebackType
import traceback
from models.util.data_preparation import *
from models.util.mongo_util import *

from models.Monitor.GeneralMonitor import GeneralMonitor
from models.Monitor.UserMonitor import UserMonitor
from models.BrainRetrieval import BrainRetrieval
from models.ChatbotRetrieval import ChatbotRetrieval
from models.Conversation import Conversation

import logging
import sys

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

MSG_PRESENTATION = "Soy avi chatbot su asistente virtual"

def thread_messengers(config: dict, brain: BrainRetrieval, others: dict, conversation: Conversation, mongoDatabase: MongoDatabase):


    if UserMonitor.verbose > 0:
        logging.info("Thread %s: starting", conversation.unique_id)

    keep_alive = True

    while keep_alive:

        while conversation.counter > 0:

            # not flood
            if UserMonitor.verbose > 1:
                logging.info(
                    f"Thread {conversation.unique_id}: counter {conversation.counter}")

            conversation.counter -= 1
            time.sleep(1)

        while conversation.is_using_message_to_queue:
            pass

        conversation.is_using_message_to_queue = True

        full_text = ""
        for message in conversation.message_queue:
            full_text += message + " "

        original_context = conversation.context
        APP_CONTEXT = config["APP_CONTEXT"]

        try:
            new_context, msg_list, require_human = brain.thinkAnswer(
                original_context, input_message=full_text, verbose=UserMonitor.brain_verbose)

            conversation.message = full_text
            conversation.context = new_context
            conversation.checkContextInputMemory(brain.chatbots, full_text)
            
            
            is_ignoring_bot = False

            if APP_CONTEXT == "walichat":
            
                api = others.get("api", None)
                if api is not None:
                    response = api.getChatDetails(others.get("device_id", ""),
                        others.get("chat_w_id", ""))
                    
                    chat_details = json.loads(response.text)
                    # print("CHAT DETAILS", chat_details)
                    
                    old_labels = chat_details.get("labels", [])

                    is_ignoring_bot = "NO_BOT" in old_labels

            old_show_bot_presentation = copy.deepcopy(conversation.show_bot_presentation)

            if require_human:
                # call this function with true to reset flag of presentation
                conversation.updateRequireHuman(require_human)
                UserMonitor.setBotError(
                    APP_CONTEXT, "bot not known what answer", others)

            
            if not is_ignoring_bot:

                if old_show_bot_presentation:
                    UserMonitor.sendMessage(
                            APP_CONTEXT, conversation.mobile, MSG_PRESENTATION, others)
                    time.sleep(Config.TIME_SECONDS_WAIT_TEXT_MESSAGE)
                    conversation.show_bot_presentation = False


                out = ""
                if Config.mode == "test" or Config.mode == "staging":
                    out = "*" + conversation.context.getLastTree() + "*"
                    out += "\n *" + conversation.context.getLastTags() + "*"
                    out += "\n *" + conversation.context.getLastProbabilities() + "*"
                # call this function with true to reset flag of presentation
                conversation.updateRequireHuman(require_human)
                if out != "":
                    UserMonitor.sendMessage(
                        APP_CONTEXT, conversation.mobile, out, others)
                    time.sleep(Config.TIME_SECONDS_WAIT_TEXT_MESSAGE * 2)




                for msg in msg_list:
                    response = None
                    if "|" in msg:
                        # message with multimedia
                        msg_split = msg.split("|")
                        msg_text = msg_split[0]
                        file_id = msg_split[1]

                        UserMonitor.sendMessageMultimedia(
                            APP_CONTEXT, conversation.mobile, msg_text, file_id, others)
                        time.sleep(Config.TIME_SECONDS_WAIT_MULTIMEDIA_MESSAGE)
                    elif "img" in msg:

                        file_id = msg

                        UserMonitor.sendMessageMultimedia(
                            APP_CONTEXT, conversation.mobile, "", file_id, others)

                        time.sleep(Config.TIME_SECONDS_WAIT_MULTIMEDIA_MESSAGE)
                    else:
                        UserMonitor.sendMessage(
                            APP_CONTEXT, conversation.mobile, msg, others)
                        time.sleep(Config.TIME_SECONDS_WAIT_TEXT_MESSAGE)

                    # logging.info(f"unique id: {unique_id}, sending message response: {response.text}")

                conversation.updateLastBotInteraction()
                UserMonitor.setBotReplied(APP_CONTEXT, others)

            count_update = UserMonitor.updateConversation(
                mongoDatabase, brain.collection, config["APP_CONTEXT"], brain.getName(), conversation)

            # print("Waiting 5...")
            # time.sleep(5)
            conversation.message_queue = []


            keep_alive = False

            if len(conversation.buffer_message_queue) > 0:
                print("COPYNG FROM BUFFER")
                conversation.message_queue = copy.deepcopy(
                    conversation.buffer_message_queue)
                # this is important beacause meanwhile we delete someone can send another message
                for buffer_message in conversation.message_queue:
                    conversation.buffer_message_queue.remove(buffer_message)

                conversation.counter = Config.TIME_SECONDS_WAIT_BASE_THREAD // 2

                keep_alive = True

            conversation.is_using_message_to_queue = False

        except Exception as exception:
            print(traceback.format_exc())

            exception_name = type(exception).__name__
            out = "DEBUG: Ha ocurrido un problema relacionado con: " + \
                str(exception) + \
                ". Asegúrese de que el enrutamiento está correctamente hecho en cada chatbot."

            if exception_name == "IndexError":
                out = "DEBUG: Ocurrió un problema, asegúrese de haber asignado respuestas y en caso de que requiriera contexto, de haber asignado respuestas de contexto"
            else:
                print("Unknown exception", exception_name)

            if Config.mode == "test" or Config.mode == "staging":

                UserMonitor.sendMessage(
                    APP_CONTEXT, conversation.mobile, out, others)
            UserMonitor.setBotError(
                APP_CONTEXT, "Error in SERVER SIDE", others)

            keep_alive = False

    conversation.changeShouldBeDeleted(True)
    if UserMonitor.verbose > 0:
        logging.info("Thread %s: finishing", conversation.unique_id)
