
import json
import pickle
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from ..ChatbotRetrieval import ChatbotRetrieval


from ..db.MongoDatabase import MongoDatabase
from ..util.data_preparation import *
from ..util.mongo_util import *

# python test_individual_model.py <nombre_modelo> <nombre_agente>


def test_individual_model(mongoDatabase: MongoDatabase, chatbots, brain_name, chatbot_name):

    ChatbotRetrieval.save_path = Config.models_saved

    chatbot: ChatbotRetrieval = ChatbotRetrieval.convertDocumentToChatbotRetrieval(brain_name, chatbot_name, chatbots[chatbot_name])

    while True:
        message = input(">>").lower()
        if message == "q" or message == "exit" or message == "quit":
            break
        probabilities, best_intent = chatbot.predict(message)

        msg = []
        if best_intent["is_natural_language_tag"]:
            msg = best_intent["responses"]
            
        next_chatbot_name = best_intent["next_chatbot_name"]
        next_chatbot_name_if_no_context = ""
        if "next_chatbot_name_if_no_context" in best_intent:
            next_chatbot_name_if_no_context = next_chatbot_name_if_no_context
        print(probabilities)
        print(msg, "pasando a", next_chatbot_name, next_chatbot_name_if_no_context)


