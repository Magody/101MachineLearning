from models.tests.test_monitor import test_monitor
from models.tests.test_individual_model import test_individual_model
from models.db.MongoDatabase import *
from models.tests.test_brain import test_brain
import sys
import os
from models.Config import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TestHelper:
    def __init__(
        self,
        mongoDatabase: MongoDatabase,
        user: str,
        brain_name: str,
        collection_name_conversations: str = "conversations",
    ):
        """Init the chatbots and database connector.

        Args:
            mongoDatabase (MongoDatabase): Mongo db connection
            user (str): user name registered on mongo db
            brain_name (str): Brain name registered on mongo db
            collection_name_conversations (str, optional): The name of collection where conversations will be stored. Defaults to "conversations".
        """
        self.mongoDatabase = mongoDatabase
        self.collection_name_conversations = collection_name_conversations
        self.set_chatbots(user, brain_name)

    def set_chatbots(self, user: str, brain_name: str) -> None:
        """Query the brain and extract the chatbots

        Args:
            user (str): user registered in mongo db
            brain_name (str): brain name related to user in mongo db
        """
        self.user = user
        self.brain_name = brain_name
        self.brain = self.mongoDatabase.find(
            "brains", {"user": user, "name": brain_name}
        )
        self.chatbots = self.brain.get("chatbots", None)

    def run_commant_test(
        self, test_type: str = "brain", chatbot_name: str = None
    ) -> None:
        """Command line tool to chat with the brain/chatbot/monitor from command line

        Args:
            test_type (str, optional): There are three types: brain(chat with all system), individual(chat with only one chatbot), monitor(check the multiple message feature). Defaults to "brain".
            chatbot_name (str, optional): In case test_type is "individual" MUST specify the chatbot_name. Defaults to None.
        """
        if test_type == "brain":
            test_brain(
                self.collection_name_conversations,
                self.mongoDatabase,
                self.brain_name,
                self.chatbots,
            )
        elif test_type == "individual":
            test_individual_model(
                self.mongoDatabase, self.chatbots, self.brain_name, chatbot_name
            )
        elif test_type == "monitor":
            test_monitor(self.mongoDatabase, self.user, self.brain_name)
        else:
            print(test_type, "does not exist")


def main():
    """App to run quick test with chatbot through the command line
    Base:    python app_test.py <test_type(brain,individual,monitor)> <brain_name> <optional:chatbot_specific>
    Example1: python app_test.py brain brain_walichat
    Example2: python app_test.py individual brain_walichat cb_signatures
    Example2: python app_test.py monitor brain_walichat
    """
    args = sys.argv
    print("Number of arguments:", len(sys.argv), "arguments.")
    print("Argument List:", str(args))

    test_type = args[1]
    brain_name = args[2]
    chatbot_name = "unknown"
    if len(args) >= 4:
        chatbot_name = args[3]

    Config.setEnvironment(os.path.join(".env.json"))

    mongoDatabase = MongoDatabase(
        Config.DB_NAME,
        f"mongodb://{Config.MONGODB_USER}:{Config.MONGODB_PASSWORD}@{Config.HOST}:{Config.MONGO_DB_PORT}/{Config.DB_NAME}?retryWrites=true&w=majority",
    )

    testHelper = TestHelper(mongoDatabase, Config.USERNAME, brain_name)
    testHelper.run_commant_test(test_type, chatbot_name)


if __name__ == "__main__":
    # Executes only if is not an import
    main()
