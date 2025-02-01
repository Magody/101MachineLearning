import math
import json
import os
from app_test import TestHelper
from models.Config import Config
from models.db.MongoDatabase import MongoDatabase
from app_train import TrainHelper


def read_json(path: str) -> dict:
    """Open and close json files. And return its contents

    Args:
        path (str): _description_

    Returns:
        dict: json as dict
    """
    f = open(path, "r")
    data = json.load(f)
    f.close()
    return data


class Demo:
    def __init__(self, user: str, brain_name: str, verbose_level: int = 1) -> None:
        """Setup the envrionment variables and open the mongo db connection"""
        self.user = user
        self.brain_name = brain_name
        self.verbose_level = verbose_level
        Config.setEnvironment(os.path.join(".env.json"))
        self.collection_name_conversations = "conversations"
        self.mongoDatabase = MongoDatabase(
            Config.DB_NAME,
            f"mongodb://{Config.MONGODB_USER}:{Config.MONGODB_PASSWORD}@{Config.HOST}:{Config.MONGO_DB_PORT}/{Config.DB_NAME}?retryWrites=true&w=majority",
        )

    def insert_data(self) -> None:
        """Insert one user and one brain with chatbot connections and train info"""
        path_demo_data = os.path.join("demo_data")
        exist_user = self.mongoDatabase.find("users", {"_id": self.user})
        result_01 = "User demo: magody already exists!"
        if not exist_user:
            user_document = {"_id": self.user, "directory": "user_test"}
            result_01 = self.mongoDatabase.insertDocument("users", user_document)

        exist_brain = self.mongoDatabase.find("brains", {"_id": self.brain_name})
        result_02 = "brain_demo already exists!"
        if not exist_brain:
            brain_document = {
                "_id": f"{self.user}_{self.brain_name}",
                "user": self.user,
                "name": self.brain_name,
                "config": {
                    "tree_broker": {"broker": {"cb00": {}, "cb01": {}}},
                    "APP_CONTEXT": "TEST",
                },
                "others": {
                    "DESCRIPTION": "Just test app"
                },
                "chatbots": {
                    "cb00": read_json(os.path.join(path_demo_data, "cb00.json")),
                    "cb01": read_json(os.path.join(path_demo_data, "cb01.json")),
                    "broker": read_json(os.path.join(path_demo_data, "broker.json")),
                },
            }
            result_02 = self.mongoDatabase.insertDocument("brains", brain_document)

        if self.verbose_level > 0:
            print("Insertion keys: ", result_01, result_02)

    def train(self, chatbot_target: str = "all") -> dict:
        """Train specified chatbots of the demo brain

        Args:
            chatbot_target (str, optional): It can train one or all chatbots. If is one specify exact name. Defaults to "all".

        Returns:
            dict: training history
        """
        trainHelper = TrainHelper(
            self.mongoDatabase, self.user, self.brain_name, self.verbose_level - 1
        )
        return trainHelper.train(chatbot_target=chatbot_target, save_plots=True)

    def test(self, test_type: str = "brain", chatbot_name: str = None) -> None:
        """Command line tool to chat with the brain/chatbot/monitor from command line

        Args:
            test_type (str, optional): There are three types: brain(chat with all system), individual(chat with only one chatbot), monitor(check the multiple message feature). Defaults to "brain".
            chatbot_name (str, optional): In case test_type is "individual" MUST specify the chatbot_name. Defaults to None.
        """
        testHelper = TestHelper(self.mongoDatabase, self.user, self.brain_name)
        testHelper.run_commant_test(test_type, chatbot_name)


user = "magody"
brain_name = "brain_demo"
chatbot_target = "all"
verbose_level = 2
test_type = "brain"
insert_initial_data = True
train = True
test = True

demo = Demo(user, brain_name, verbose_level)

if insert_initial_data:
    if verbose_level > 0:
        print("Beginning insertion of demo data in mongodb...")
    demo.insert_data()
if train:
    if verbose_level > 0:
        print("Beginning training...")
    history = demo.train(chatbot_target)

if test:
    if verbose_level > 0:
        print("Beginning terminal test...")
    demo.test(test_type=test_type)
