import os

from numpy.lib.function_base import average
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from models.ChatbotRetrieval import ChatbotRetrieval
import random
import json
import pickle
import numpy as np
import sys
from datetime import datetime, date
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

from models.db.MongoDatabase import MongoDatabase
from models.util.data_preparation import *
from models.util.mongo_util import *

class TrainHelper:
    
    def __init__(self, mongoDatabase: MongoDatabase, user:str, brain_name:str, verbose:int=1): 
        """Setup database and query user/brain info.

        Args:
            mongoDatabase (MongoDatabase): Database connection object to mongo db
            user (str): user that has the brain
            brain_name (str): name of the brain in the database
            verbose (int, optional): Verbose level for log messages. Defaults to 1.
        """
        self.user = user
        self.brain_name = brain_name
        self.verbose = verbose
        self.mongoDatabase = mongoDatabase
        self.path_brain_flow = os.path.join("chatbots", "brain_flow")
        self.get_brain_info()
        # Set the static save_path for pickle and other files models
        ChatbotRetrieval.save_path = os.path.join("chatbots", "models_saved")
        self.history = {}
        
    def get_brain_info(self):
        """Query/Update the brain and chatbot info from database. Also updates the chain TXT files
        for training.
        """
        self.brain = self.mongoDatabase.find("brains", {"user": self.user, "name": self.brain_name})
        self.chatbots = self.brain.get("chatbots", None)
        self.tree_broker = self.brain["config"]["tree_broker"]
        self.generate_chatbot_chain_training()
        

    def generate_chatbot_chain_training(self):
        """Generate TXT files using the training info for each chatbot. Some chatbots
        only predicts the path, so these chatbots needs a txt with a compilation of all
        data for every path. Also set all the chatbots training intents based on TXT generated.
        """
        generateConcatsFromArrays(self.path_brain_flow, self.tree_broker, self.chatbots, self.verbose-1)
        self.chatbots_ready_to_train: dict = fill_non_natural_language_intents(self.path_brain_flow, self.chatbots, self.verbose-1)

    def update_history(self, history:dict, acc_test:list)->None:
        """Updates history based on model tensorflow history and an array with test
        accuracy metrics

        Args:
            history (dict): ML NN model fit history result
            acc_test (list): array with loss and accuracy for testing
        """
        self.history["train_loss"] = history['loss']
        self.history["train_categorical_accuracy"] = history['categorical_accuracy']
        self.history["val_loss"] = history['val_loss']
        self.history["val_categorical_accuracy"] = history['val_categorical_accuracy']
        self.history["test_lose"].append(acc_test[0])
        self.history["test_acc"].append(acc_test[1])
        self.history["train_lose"].append(self.history["train_loss"][-1])
        self.history["train_acc"].append(self.history["train_categorical_accuracy"][-1])
        self.history["val_lose"].append(self.history["val_loss"][-1])
        self.history["val_acc"].append(self.history["val_categorical_accuracy"][-1])
    
    def save_history_plots(self, key:str, chatbot: ChatbotRetrieval)->None:
        #print(history)
        plt.figure(key)
        plt.subplot(1, 2, 1)
        plt.plot(range(1, chatbot.epochs+1), self.history["train_loss"], label="Train loss")
        plt.plot(range(1, chatbot.epochs+1), self.history["val_loss"], label="Val loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, chatbot.epochs+1), self.history["train_categorical_accuracy"], label="Train accuracy")
        plt.plot(range(1, chatbot.epochs+1), self.history["val_categorical_accuracy"], label="Val accuracy")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Categorical accuracy')
        plt.legend()
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(ChatbotRetrieval.save_path, self.brain_name, key, f"summary_{dt_string}.png"))
        

    def train(self, chatbot_target:str="all", evaluate_weakness:bool=False, save_plots:bool=False)->dict:
        """Trains N chatbots based on types. Also generates training history vars

        Args:
            chatbot_target (str, optional): It can train only one chatbot or all chatbots in the brain. Defaults to "all"
            evaluate_weakness (bool, optional): Create an isolated test data to check training without all words. Defaults to False.
            save_plots (bool, optional): Save *.png images with some history vars for every Chatbot. Defaults to False.

        Returns:
            dict: history data for training of each chatbot.
        """
        self.history = {
            "scores": {},
            "test_lose": [],
            "test_acc": [],
            "train_lose": [],
            "train_acc": [],
            "val_lose": [],
            "val_acc": []
        }
        

        for key, document_chatbot in self.chatbots_ready_to_train.items():


            if key == "human":
                continue

            if chatbot_target != "all" and chatbot_target != key:
                continue

            if self.verbose > 0:
                print("Training", key)
                
            
            chatbot: ChatbotRetrieval = ChatbotRetrieval.convertDocumentToChatbotRetrieval(self.brain_name, key, document_chatbot)
            # chatbot.epochs = 50
            # chatbot.batch_size = 4
            h, acc_test = chatbot.train(porc_train=1, porc_val=0, verbose=(self.verbose-1))
            # print(verbose, "verbose", type(verbose))
            self.update_history(h, acc_test)
            
            if evaluate_weakness:
                if self.verbose > 0:
                    print("Evaluating weakness")
                scores = chatbot.evaluateWeakness(self.verbose-1)
                print(key, scores)
                self.history["scores"][key] = scores

            if self.verbose > 0 and save_plots:
                self.save_history_plots(key, chatbot)
        
        # Final logs
        if self.verbose > 0:
            print("\n\n", self.history["scores"])
            print("*****")
            train_acc = round(average(self.history["train_acc"])*100, 2)
            test_acc = round(average(self.history["test_acc"])*100, 2)
            val_acc = round(average(self.history["val_acc"])*100, 2)
            # print(f"Error medio: {average(history_test_lose)}")
            print(f"Accuracy mean: {train_acc} %")
            # print(f"Accuracy final: {history_test_acc[-1]}")

        return self.history




def main():
    """Uses command line args as input.
    Base   : python app_train.py <db_name>    <user> <brain_name>   <optional:chatbot_name|all> <optional:verbose_level>
    Example: python app_train.py chatbot_test test   brain_telegram cb_faq_firma       1              
    """
    
    arg_list = sys.argv
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    # print('Argument List:', str(arg_list))

    Config.setEnvironment(os.path.join(".env.json"))

    user = arg_list[1]
    brain_name = arg_list[2]
    chatbot_target = "all"
    if len(arg_list) >= 4:
        chatbot_target = arg_list[3]
            
    verbose = 0
    if len(arg_list) >= 5:
        verbose = int(arg_list[4])
    
    mongoDatabase = MongoDatabase(
        Config.DB_NAME,
        f"mongodb://{Config.MONGODB_USER}:{Config.MONGODB_PASSWORD}@{Config.HOST}:{Config.MONGO_DB_PORT}/{Config.DB_NAME}?retryWrites=true&w=majority",
    )
    trainHelper = TrainHelper(mongoDatabase, user, brain_name, verbose)
    trainHelper.train(chatbot_target=chatbot_target, save_plots=True)

if __name__ == "__main__":
    # Run only if wasn't imported
    main()




