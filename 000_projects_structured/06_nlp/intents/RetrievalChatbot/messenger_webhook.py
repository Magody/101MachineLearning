from models.Monitor.UserMonitor import UserMonitor
from models.Monitor.GeneralMonitor import GeneralMonitor
from models.util.mongo_util import *
from models.API import *
import requests
import sys
from flask import Flask, request, jsonify
from thread_messengers import thread_messengers
from models.db.MongoDatabase import *
from models.util.data_preparation import *
from models.BrainRetrieval import BrainRetrieval
from models.util.nltk_transforms import *
from models.Conversation import *
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# this is more flexible, Config.USERNAME should be unique and accept routing of all request for this user

# This webhook configuration supports:
# - walichat
# - test

# console configuration: python messenger_webhook.py gossip

arg_list = sys.argv
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(arg_list))


Config.setEnvironment(os.path.join(".env.json"))

mode = arg_list[1]

API_URL = f"/api/v1/chatbot/retrieval/server/1/{Config.USERNAME.replace('.', '-')}/<brain_name>"

app = Flask(__name__)


@app.route(API_URL, methods=['POST'])
def router(brain_name):

    # to-do: auth stuff with Config.USERNAME and id_brain, caché management for optimizing

    # messenger configuration

    content = request.json
    print(content)

    if isinstance(content, list):
        print("Evento de mensaje")
        return jsonify({
            "action": "es evento de tipo lista"
        })

    if content["event"] != "message:in:new":
        print("no es un evento de mensaje de entrada")
        return jsonify({
            "action": "es evento de tipo objeto"
        })

    # ignore groups

    if content["data"]["chat"]["type"] == "group":
        print("IS A GROUP, IGNORE GROUPS")
        return jsonify({
            "action": "es grupo"
        })

    unique_id = content["data"]["fromNumber"]
    mobile = content["data"]["fromNumber"]
    message = content['data']["body"]
    device_id = content["device"]["id"]
    chat_w_id = content['data']["chat"]["id"]

    type = content['data']["type"]

    others = dict()
    others["device_id"] = device_id
    others["chat_w_id"] = chat_w_id

    # general
    # database is passed, because initially this app should manage multiple users in the same folder
    # but due time, the app will be packaged for each user, so its not necesary multiple users
    # ensure the database name is parameticed
    mongoDatabase = MongoDatabase(Config.DB_NAME, "mongodb://%s:%s@%s:%d" % (Config.MONGODB_USER, Config.MONGODB_PASSWORD, Config.HOST, Config.MONGO_DB_PORT))
        

    GeneralMonitor.addMessageToUserMonitor(
        Config.USERNAME, brain_name, mobile, message, others, mongoDatabase, type)

    return jsonify({
        "conversation_id": unique_id,
        "message_response": "wait for the brain please"
    })


if __name__ == '__main__':
    try:

        GeneralMonitor.init(thread_messengers, mode)

        app.run(host='0.0.0.0', port=Config.APP_PORT, debug=True)

    except Exception as error:
        print("error", error)
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
