import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from tensorflow.keras import models
from tensorflow.python.util.dispatch import dispatch
import copy
import traceback
import threading

from models.Monitor.GeneralMonitor import GeneralMonitor
from models.Monitor.UserMonitor import UserMonitor
from flask_cors import CORS, cross_origin
# from flask_socketio import send, emit


from models.ChatbotRetrieval import ChatbotRetrieval
import random
import json
import pickle
import numpy as np

from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify

from models.util.nltk_transforms import *
from models.BrainRetrieval import BrainRetrieval

from models.util.data_preparation import *
import sys

from models.Conversation import *
from models.db.MongoDatabase import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from thread_messengers import thread_messengers
from flask_socketio import SocketIO

app = Flask(__name__)
# cors = CORS(app, resources={r"/socket.io": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'ridetel.secret.2021*'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# the chatbot telegram set the brain with program execution

# python messenger_io_websockets.py test

# console configuration
arg_list = sys.argv
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(arg_list))

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

MSG_PRESENTATION = "Soy avi chatbot su asistente virtual"

Config.setEnvironment(os.path.join(".env.json"))


mode = arg_list[1]

global namespace
namespace = f'/server/1/{Config.USERNAME}'

    
@socketio.on('bot_message', namespace=namespace)
def handle_message(message_json):
    message_dict: dict = json.loads(message_json)

    mongoDatabase = MongoDatabase(Config.DB_NAME, "mongodb://%s:%s@%s:%d" % (Config.MONGODB_USER, Config.MONGODB_PASSWORD, Config.HOST, Config.MONGO_DB_PORT))
    
    brain_name = message_dict.get("brain_name", "unknown_brain")

    global namespace
    others = dict()
    others["updater"] = socketio.emit
    others["namespace"] = namespace
    others["device_id"] = ""
    others["chat_w_id"] = ""
    others["is_testing_socket_from_different_APP_CONTEXT"] = True
    
    unique_id = message_dict.get("clientUniqueId", "unknown_user")
    mobile = unique_id
    message: str = message_dict.get("message", "unknownw_w_w")

    type = "text"


    if message.lower() == "reset":
        GeneralMonitor.resetContextOfUserBrainConversation(Config.USERNAME, mongoDatabase, "telegram", brain_name, mobile)
        print("RESETING")
        socketio.emit("bot_message", f"Si la conversación no estaba en memoria entonces el contexto para la conversación id={unique_id} se reseteó, caso contrario inténtelo luego de unos minutos si es fase de test", namespace=namespace)
        return

    ###########
    # Generic configuration
    GeneralMonitor.addMessageToUserMonitor(Config.USERNAME, brain_name, mobile, message, others, mongoDatabase, type)
        

def main() -> None:
    print("Open connection to", namespace)
    socketio.run(app, host='0.0.0.0', port=Config.SOCKET_PORT)






def thread_websocket(config: dict, brain: BrainRetrieval, others: dict, conversation: Conversation, mongoDatabase: MongoDatabase):

    with app.test_request_context("/"):
        
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
                    ". Asegúrese de que el enrutamiento está correctamente hecho en cada chatbot y que se encuentra entrenado"

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



if __name__ == '__main__':
    try:
        GeneralMonitor.init(thread_websocket, mode)
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
