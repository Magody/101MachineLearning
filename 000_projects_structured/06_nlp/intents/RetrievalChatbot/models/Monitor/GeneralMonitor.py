from models.BrainRetrieval import BrainRetrieval
from models.db.MongoDatabase import MongoDatabase
from ..Conversation import Conversation, ConversationContext
from ..Config import Config
import logging
import threading
import time
import random
import copy
from .UserMonitor import UserMonitor, test_monitor


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


class GeneralMonitor:

    # Shared memory
    alive = True
    users_monitors = {}  # indexed by username
    thread_function: threading.Thread = None
    thread_memory: threading.Thread = None
    thread_update: threading.Thread = None
    verbose = 3

    @staticmethod
    def init(thread_function: threading.Thread = test_monitor, mode=Config.MODE_TEST):
        GeneralMonitor.thread_function = thread_function
        GeneralMonitor.thread_memory = threading.Thread(
            target=thread_general_monitor_manage_memory, args=()
        )
        GeneralMonitor.thread_memory.start()
        GeneralMonitor.thread_update = threading.Thread(
            target=thread_update_data, args=()
        )
        GeneralMonitor.thread_update.start()

        Config.mode = mode

        if mode == Config.MODE_TEST:

            Config.TIME_SECONDS_WAIT_FOR_FREE_MEMORY = 10
            Config.TIME_SECONDS_WAIT_FOR_USER_MONITOR_FREE_MEMORY = 10
            Config.TIME_SECONDS_WAIT_BASE_THREAD = 5
            Config.TIME_MINUTES_BOT_WAIT_TO_RESET = 1
            Config.TIME_SECONDS_UPDATE_DATA = 10

        elif mode == Config.MODE_STAGING:
            Config.TIME_MINUTES_BOT_WAIT_TO_RESET = 3
            Config.TIME_SECONDS_UPDATE_DATA = 60

        elif mode == Config.MODE_GOSSIP:

            Config.TIME_SECONDS_WAIT_FOR_FREE_MEMORY = 10
            Config.TIME_SECONDS_WAIT_FOR_USER_MONITOR_FREE_MEMORY = 10
            Config.TIME_SECONDS_WAIT_BASE_THREAD = 13
            Config.TIME_MINUTES_BOT_WAIT_TO_RESET = 1
            Config.TIME_SECONDS_UPDATE_DATA = 10

        elif mode == Config.MODE_PRODUCTION:
            GeneralMonitor.verbose = 0
            UserMonitor.brain_verbose = 0

    @staticmethod
    def die():
        GeneralMonitor.alive = False
        users_to_remove = []
        for username, user_monitor_value in GeneralMonitor.users_monitors.items():
            users_to_remove.append(username)

        for username in users_to_remove:
            GeneralMonitor.removeUserMonitorFromMemory(username)

        logging.info(f"Closing monitor")

    @staticmethod
    def existUserMonitor(username: str) -> bool:
        return username in GeneralMonitor.users_monitors

    @staticmethod
    def addUserMonitor(username: str, mongoDatabase: MongoDatabase) -> bool:

        if GeneralMonitor.existUserMonitor(username):

            if not GeneralMonitor.users_monitors[username]["is_being_deleted"]:
                print("unique id already exists in shared memory")
                return False

            else:
                # wait while is being deleted
                while GeneralMonitor.users_monitors[username]["is_being_deleted"]:
                    pass

        GeneralMonitor.users_monitors[username] = dict()
        GeneralMonitor.users_monitors[username]["user_monitor"] = UserMonitor(
            GeneralMonitor.thread_function, mongoDatabase, username
        )
        GeneralMonitor.users_monitors[username]["is_being_deleted"] = False

        return True

    @staticmethod
    def existMessagesToProcess() -> bool:

        for username, user_monitor_dict in GeneralMonitor.users_monitors.items():

            userMonitor: UserMonitor = user_monitor_dict["user_monitor"]

            for brain_name, brain_dict in userMonitor.brain_conversations.items():

                conversations: dict = brain_dict["conversations"]

                for unique_id, value in conversations.items():

                    conversation: Conversation = value

                    if (
                        len(conversation.message_queue) > 0
                        or conversation.is_using_message_to_queue
                        or len(conversation.buffer_message_queue) > 0
                    ):
                        # exist messages
                        return True

        return False

    @staticmethod
    def addMessageToUserMonitor(
        username: str,
        brain_name: str,
        mobile: str,
        message: str,
        others: dict,
        mongoDatabase: MongoDatabase,
        message_type: str,
    ) -> bool:

        if not GeneralMonitor.existUserMonitor(username):

            if not GeneralMonitor.addUserMonitor(username, mongoDatabase):
                print("Error: Can't create the UserMonitor")
                return False

        unique_id = f"{username}_{brain_name}_{mobile}"
        if not GeneralMonitor.users_monitors[username][
            "user_monitor"
        ].addMessageToQueue(
            brain_name, unique_id, mobile, message, others, mongoDatabase, message_type
        ):
            print(
                "General Monitor: addMessageToUserMonitor: No se pudo agregar mensaje a ",
                username,
                brain_name,
                unique_id,
            )

        return True

    @staticmethod
    def removeUserMonitorFromMemory(username: str) -> bool:

        if not GeneralMonitor.existUserMonitor(username):
            print("Cant remove from memory, something not exist")
            return False

        GeneralMonitor.users_monitors[username]["user_monitor"].alive = False
        GeneralMonitor.users_monitors[username]["is_being_deleted"] = True

        del GeneralMonitor.users_monitors[username]
        # is not necesary verify if is being deleted is false

        # if return True, mean all is ok
        return not GeneralMonitor.existUserMonitor(username)

    @staticmethod
    def resetContextOfUserBrainConversation(
        username, mongoDatabase, APP_CONTEXT, brain_name, mobile
    ):

        userMonitor: UserMonitor = GeneralMonitor.users_monitors.get(username, None)
        if userMonitor is None:
            userMonitor: UserMonitor = UserMonitor(None, mongoDatabase, username)
        else:
            userMonitor = userMonitor["user_monitor"]
        userMonitor.resetContextOfBrainConversation(
            APP_CONTEXT, brain_name, f"{username}_{brain_name}_{mobile}"
        )


def thread_general_monitor_manage_memory():

    while True:

        if not GeneralMonitor.alive:
            return
        time.sleep(1)

        users_to_remove = []

        for username, user_monitor_dict in GeneralMonitor.users_monitors.items():

            userMonitor: UserMonitor = user_monitor_dict.get("user_monitor", None)
            if userMonitor is None:
                continue

            if userMonitor.should_be_deleted:
                # try to free space
                while userMonitor.is_updating_time_to_live:
                    pass

                userMonitor.is_updating_time_to_live = True

                userMonitor.time_to_live -= 1

                ttl = userMonitor.time_to_live

                # not flood
                if GeneralMonitor.verbose > 1 and Config.mode == Config.MODE_TEST:
                    logging.info(f"username {username}: time_to_live {ttl}")

                if userMonitor.time_to_live <= 0:
                    users_to_remove.append(username)

                userMonitor.is_updating_time_to_live = False

        for username in users_to_remove:

            userMonitor: UserMonitor = GeneralMonitor.users_monitors[username][
                "user_monitor"
            ]

            # doble safe control
            if userMonitor.should_be_deleted:
                while userMonitor.is_updating_time_to_live:
                    pass

                if userMonitor.time_to_live <= 0:

                    length_conversations = len(GeneralMonitor.users_monitors.items())
                    logging.info(
                        f"Trying to remove User monitor {username}: actual length: {length_conversations}"
                    )
                    if GeneralMonitor.removeUserMonitorFromMemory(username):
                        length_conversations = len(
                            GeneralMonitor.users_monitors.items()
                        )
                        logging.info(
                            f"username {username}: removed, actual length: {length_conversations}"
                        )
                    else:
                        logging.info(
                            "username %s: Error: cant remove from memory the unique ID",
                            username,
                        )


def thread_update_data():

    while True:

        try:
            # print("Updating from database")
            for username, user_monitor_dict in GeneralMonitor.users_monitors.items():
                user_monitor_dict["user_monitor"].reloadBrains()
                user_monitor_dict["user_monitor"].reloadModelsInAllBrains()
                print("reloaded models")
                print("user_monitors", len(GeneralMonitor.users_monitors.items()))
        except Exception as exception:
            pass

        for second in range(Config.TIME_SECONDS_UPDATE_DATA):
            if not GeneralMonitor.alive:
                return
            time.sleep(1)
