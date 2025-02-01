from requests.models import Response
from ..API import APIWaliChat
from ..db.MongoDatabase import MongoDatabase
from ..util.TimeUtil import TimeUtil
from ..ChatbotRetrieval import ChatbotRetrieval
from ..Conversation import Conversation, ConversationContext
from ..BrainRetrieval import BrainRetrieval
from ..Config import Config
import logging
import threading
import time
import random
import copy
import json
from ..custom_bussiness import (
    customManageMemory,
    customManageState,
    customManageContextMessage,
    customManageHuman,
)


class UserMonitor:
    # indexed by Monitor by unique username

    verbose = 3
    brain_verbose = 0

    def __init__(
        self,
        thread_function: threading.Thread,
        mongoDatabase: MongoDatabase,
        username: str,
    ):
        self.brain_conversations = dict()
        self.should_be_deleted = False
        self.thread_function = thread_function
        self.resetTimeToLive()
        self.alive = True
        self.brains = dict()
        self.mongoDatabase = mongoDatabase
        self.username = username

        self.reloadBrains()

        self.thread_memory = threading.Thread(
            target=thread_user_monitor_manage_memory, args=(self,)
        )
        self.thread_memory.start()

    def reloadBrains(self):
        brains_list: list = self.mongoDatabase.find_all(
            "brains", {"user": self.username}
        )
        # print("BRAIN LIST", len(brains_list), self.username)

        for brain in brains_list:
            id = brain["_id"]
            self.brains[id] = brain
            # print("BRAIN",id)

    def reloadModelsInAllBrains(self):
        for brain_code, brain_conversation_object in self.brain_conversations.items():
            brain: BrainRetrieval = brain_conversation_object["model"]
            brain.reloadModels()

    @staticmethod
    def updateConversation(
        mongoDatabase: MongoDatabase,
        conversations_collection,
        APP_CONTEXT,
        unique_id,
        conversation: Conversation,
    ):

        return mongoDatabase.update(
            conversations_collection,
            {"_id": conversation.unique_id},
            conversation.getDocumentForMongoDB(),
        )

    def existBrain(self, brain: str) -> bool:
        return brain in self.brain_conversations

    def createBrain(self, brain_name: str):
        brain_code = f"{self.username}_{brain_name}"
        if brain_code not in self.brains:
            print(f"Brain code {brain_code} not exist in known brains")
            return False

        self.brain_conversations[brain_name] = dict()
        self.brain_conversations[brain_name]["config"] = self.brains[brain_code][
            "config"
        ]

        collection = "conversations"
        self.brain_conversations[brain_name]["model"] = BrainRetrieval(
            brain_name,
            self.brains[brain_code]["chatbots"],
            customManageMemory,
            customManageState,
            customManageContextMessage,
            customManageHuman,
        )

        # this are GENERAL OTHERS
        self.brain_conversations[brain_name]["others"] = self.brains[brain_code][
            "others"
        ]

        if self.brain_conversations[brain_name]["config"]["APP_CONTEXT"] == "walichat":
            mode = "test" if Config.mode == "gossip" else Config.mode
            token = self.brain_conversations[brain_name]["others"]["WALI_CHAT"][
                "TOKEN"
            ][mode]
            self.brain_conversations[brain_name]["others"]["api"] = APIWaliChat(token)

        # we cant combine general others with specific others,
        #  we can optimice only referencing to others ins the first
        # brains queried self.brain_conversations[brain_name]["others"].update(others)

        self.brain_conversations[brain_name]["conversations"] = dict()
        return True

    def existUniqueIdConversationInBrain(self, brain: str, unique_id: str) -> bool:

        if self.existBrain(brain):
            try:
                return unique_id in self.brain_conversations[brain]["conversations"]
            except:
                return False

        return False

    def addConversationToBrain(self, brain: str, conversation: Conversation):

        unique_id = conversation.unique_id

        if self.existUniqueIdConversationInBrain(brain, unique_id):

            if not self.brain_conversations[brain]["conversations"][
                unique_id
            ].is_being_deleted:
                print("unique id ya existe en la memoria compartida")
                return

            else:
                # wait while is being deleted
                while self.brain_conversations[brain]["conversations"][
                    unique_id
                ].is_being_deleted:
                    pass

        self.brain_conversations[brain]["conversations"][unique_id] = conversation
        self.brain_conversations[brain]["conversations"][
            unique_id
        ].changeShouldBeDeleted(False)

    def resetContext(self, brain: str, unique_id: str):
        if self.existUniqueIdConversationInBrain(brain, unique_id):
            self.brain_conversations[brain]["conversations"][unique_id][
                "conversation"
            ].resetContext()
            return True
        else:
            return False

    def removeConversationFromMemory(self, brain: str, unique_id: str) -> bool:

        if not self.existUniqueIdConversationInBrain(brain, unique_id):
            print("Cant remove from memory, something not exist")
            return False
        self.brain_conversations[brain]["conversations"][
            unique_id
        ].is_being_deleted = True

        del self.brain_conversations[brain]["conversations"][unique_id]
        # is not necesary verify if is being deleted is false

        # if return True, mean all is ok
        return not self.existUniqueIdConversationInBrain(brain, unique_id)

    def isThreadCreated(self, brain: str, unique_id: str):

        if self.existUniqueIdConversationInBrain(brain, unique_id):
            conversation: Conversation = self.brain_conversations[brain][
                "conversations"
            ][unique_id]
            while conversation.is_creating_thread:
                pass

            return conversation.is_thread_created
        else:
            return False

    def createThread(self, brain: str, unique_id: str, final_others: dict) -> bool:
        # final_others: GENERAL OTHERS + SPECIFIC USER CONVERSATION OTHERS
        if self.thread_function is None:
            print(
                "No se ha definido la función thread_function en GeneralMonitor -> UserMonitor"
            )
            return False

        if self.isThreadCreated(brain, unique_id):
            print("Ya existe un thread creado para", brain, unique_id)
            return False

        if self.existUniqueIdConversationInBrain(brain, unique_id):

            conversation: Conversation = self.brain_conversations[brain][
                "conversations"
            ][unique_id]

            if conversation.is_creating_thread:
                print("Ya se está creando el hilo para ", unique_id)
                return False
            else:

                conversation.is_creating_thread = True
                logging.info("Creating thread for " + unique_id)
                x = threading.Thread(
                    target=self.thread_function,
                    args=(
                        self.brain_conversations[brain]["config"],
                        self.brain_conversations[brain]["model"],
                        final_others,
                        conversation,
                        self.mongoDatabase,
                    ),
                )

                logging.info("Before running thread " + unique_id)
                x.start()

                conversation.thread = x
                conversation.is_creating_thread = False

                conversation.is_thread_created = True
                return True
        else:
            return False

    def resetTimeToLive(self):
        self.is_updating_time_to_live = True
        self.time_to_live = Config.TIME_SECONDS_WAIT_FOR_USER_MONITOR_FREE_MEMORY
        self.is_updating_time_to_live = False

    def changeShouldBeDeleted(self, condition: bool):
        self.should_be_deleted = condition
        if not condition:
            self.resetTimeToLive()

    def resetContextOfBrainConversation(self, APP_CONTEXT, brain_name, unique_id):

        conversations_collection = "conversations"
        if brain_name in self.brain_conversations:
            conversations_collection = self.brain_conversations[brain_name][
                "model"
            ].collection

        conversation_query = self.mongoDatabase.find(
            conversations_collection, {"_id": unique_id}
        )

        if conversation_query:
            conversation = Conversation.convertMongoDBDocumentToObject(
                conversation_query
            )
            conversation.resetContext()

            count_update = UserMonitor.updateConversation(
                self.mongoDatabase,
                conversations_collection,
                APP_CONTEXT,
                brain_name,
                conversation,
            )

            if self.existUniqueIdConversationInBrain(brain_name, unique_id):
                del self.brain_conversations[brain_name]["conversations"][unique_id]
        else:
            print(
                "No se pudo resetear el contexto", unique_id, conversations_collection
            )

    def addMessageToQueue(
        self,
        brain_name: str,
        unique_id: str,
        mobile: str,
        message: str,
        specific_others: dict,
        mongoDatabase: MongoDatabase,
        type: str,
    ) -> bool:

        if not self.existBrain(brain_name):
            if not self.createBrain(brain_name):
                print("No se pudo crear el brain", brain_name, unique_id)
                return False

        GENERAL_OTHERS = self.brain_conversations[brain_name]["others"]

        final_others: dict = copy.deepcopy(GENERAL_OTHERS)
        final_others.update(specific_others)

        APP_CONTEXT = self.brain_conversations[brain_name]["config"].get(
            "APP_CONTEXT", ""
        )
        conversations_collection = self.brain_conversations[brain_name][
            "model"
        ].collection

        if not self.existUniqueIdConversationInBrain(brain_name, unique_id):
            # no exist conversation in brain, it is not the first message in brain

            APP_CONTEXT = self.brain_conversations[brain_name]["config"]["APP_CONTEXT"]

            conversation_query = mongoDatabase.find(
                conversations_collection, {"_id": unique_id}
            )

            if conversation_query:
                # conversation already exists
                conversation = Conversation.convertMongoDBDocumentToObject(
                    conversation_query
                )
                conversation.device_id = final_others.get("device_id", "")
                conversation.chat_w_id = final_others.get("chat_w_id", "")
            else:
                # save new phone
                conversation = Conversation(
                    unique_id,
                    mobile,
                    ConversationContext("broker"),
                    final_others.get("device_id", ""),
                    final_others.get("chat_w_id", ""),
                    name=final_others.get("name", ""),
                    username=self.username,
                    app_context=APP_CONTEXT,
                    brain_name=brain_name,
                )
                conversation.updateFirstClientInteraction()
                id = mongoDatabase.insertDocument(
                    conversations_collection, {"_id": unique_id}
                )
                if id != "":
                    count_update = UserMonitor.updateConversation(
                        self.mongoDatabase,
                        conversations_collection,
                        APP_CONTEXT,
                        unique_id,
                        conversation,
                    )

                    print(count_update == 1)
                else:
                    print("No se pudo insertar")
                    return False

            conversation.updateLastClientInteraction()
            conversation.message = message

            self.addConversationToBrain(brain_name, conversation)

        conversation: Conversation = self.brain_conversations[brain_name][
            "conversations"
        ][unique_id]

        if type != "text":
            conversation.updateRequireHuman(True)
            message = type
            UserMonitor.setBotError(APP_CONTEXT, message, final_others)

        if conversation.require_human:

            if not conversation.allowCommunicationEvenHumanRequest():

                conversation.context.addHistory(message, [], [], [], False)

                count_update = UserMonitor.updateConversation(
                    self.mongoDatabase,
                    conversations_collection,
                    APP_CONTEXT,
                    brain_name,
                    conversation,
                )

                return False

            elif type != "text":

                human_responses: dict = self.brain_conversations[brain_name]["tags"]

                msg_human = human_responses["message_human_in_laboral_time"][0]

                if not TimeUtil.isLaboralTime(1):
                    msg_human = human_responses["message_human_out_laboral_time"][0]

                UserMonitor.sendMessage(
                    APP_CONTEXT, conversation.mobile, message, final_others
                )

                conversation.context.addHistory(type, [], [], [msg_human])
                count_update = UserMonitor.updateConversation(
                    self.mongoDatabase,
                    conversations_collection,
                    APP_CONTEXT,
                    brain_name,
                    conversation,
                )

                print("IS NOT TEXT, REQUIRE HUMAN", unique_id, message)
                return False

        if not self.isThreadCreated(brain_name, unique_id):
            print("No existe un thread creado para", unique_id, "intentando crearlo")

            if not self.createThread(brain_name, unique_id, final_others):
                print("No se pudo crear el thread para", unique_id)
                return False

        print(
            "INCOMING NEW MESSAGE: ",
            message,
            "is_using_message_to_queue",
            conversation.is_using_message_to_queue,
        )
        if conversation.is_using_message_to_queue:
            # store in buffer better than using another thread
            if (
                message not in conversation.buffer_message_queue
                and message not in conversation.message_queue
            ):
                conversation.buffer_message_queue.append(message)
            else:
                return True
        else:
            conversation.is_using_message_to_queue = True

            if message not in conversation.message_queue:
                conversation.message_queue.append(message)
            else:
                conversation.is_using_message_to_queue = False
                return True

            conversation.counter = Config.TIME_SECONDS_WAIT_BASE_THREAD

            conversation.is_using_message_to_queue = False

        conversation.changeShouldBeDeleted(False)
        self.changeShouldBeDeleted(False)

        # buffer keeps alive the thread if exist, but if thread is not alive and receive a message
        # should reinit the thread
        if not conversation.thread.is_alive():
            conversation.thread = None
            conversation.is_thread_created = False

            if not self.createThread(brain_name, unique_id, final_others):
                print("No se pudo REINICIAR el thread para", unique_id)

            conversation.counter = Config.TIME_SECONDS_WAIT_BASE_THREAD

        UserMonitor.setBotActive(APP_CONTEXT, final_others)
        return True

    @staticmethod
    def sendMessage(APP_CONTEXT: str, mobile: str, message: str, others: dict):
        if message == "":
            return

        if others.get("is_testing_socket_from_different_APP_CONTEXT", False):
            emit = others.get("updater", None)
            namespace = others.get("namespace", "/")
            if emit is not None:
                print("Emit", message, "namespace", namespace)
                emit("bot_message", message, namespace=namespace)
            else:
                print("Emit is None, cant send message")
            return

        if APP_CONTEXT == "walichat":
            api: APIWaliChat = others.get("api", None)
            if Config.mode != Config.MODE_GOSSIP:
                api.sendMessage(mobile, message)
        elif APP_CONTEXT == "telegram":
            chat_w_id = others.get("chat_w_id", "")

            updater = others.get("updater", None)
            updater.bot.sendMessage(
                chat_id=chat_w_id,
                parse_mode="HTML",
                text=message.replace("<", "").replace(">", ""),
            )
        elif APP_CONTEXT == "test":
            print(message)

    @staticmethod
    def sendMessageMultimedia(
        APP_CONTEXT: str, mobile: str, message: str, file_id: str, others: dict
    ):
        if message == "":
            return

        if others.get("is_testing_socket_from_different_APP_CONTEXT", False):
            emit = others.get("updater", None)
            namespace = others.get("namespace", "/")
            if emit is not None:
                print("Emit", message, "namespace", namespace)
                emit("bot_message", message, namespace=namespace)
            else:
                print("Emit is None, cant send message")
            return

        if APP_CONTEXT == "walichat":
            api: APIWaliChat = others.get("api", None)
            id_multimedia = others["WALI_CHAT"]["FILES_IDS"][file_id]
            if Config.mode != Config.MODE_GOSSIP:
                print(api.sendMessageMultimedia(mobile, message, id_multimedia))
        elif APP_CONTEXT == "telegram":
            chat_w_id = others.get("chat_w_id", "")

            updater = others.get("updater", None)
            updater.bot.sendMessage(
                chat_id=chat_w_id,
                parse_mode="HTML",
                text=message.replace("<", "").replace(">", ""),
            )
        elif APP_CONTEXT == "test":
            print(message)

    @staticmethod
    def setBotActive(APP_CONTEXT: str, others: dict):

        if APP_CONTEXT == "walichat":
            api: APIWaliChat = others.get("api", None)

            response: Response = api.getChatDetails(
                others.get("device_id", ""), others.get("chat_w_id", "")
            )

            chat_details = json.loads(response.text)
            # print("CHAT DETAILS", chat_details)

            old_labels = chat_details.get("labels", [])
            old_labels.extend(["bot_error"])

            print(
                api.assignLabels(
                    others.get("device_id", ""), others.get("chat_w_id", ""), old_labels
                ).text
            )

            if Config.mode != Config.MODE_GOSSIP:
                print(
                    api.assignStatus(
                        others.get("device_id", ""),
                        others.get("chat_w_id", ""),
                        APIWaliChat.STATUS_ACTIVE,
                    ).text
                )

    @staticmethod
    def setBotError(APP_CONTEXT: str, note: str, others: dict):

        if Config.mode == Config.MODE_TEST:
            print("TEST, set bot error", note)
        if APP_CONTEXT == "walichat":
            api: APIWaliChat = others.get("api", None)

            print(
                api.createNote(
                    others.get("device_id", ""),
                    others.get("chat_w_id", ""),
                    "El bot falló. Razón: " + note,
                ).text
            )

            response: Response = api.getChatDetails(
                others.get("device_id", ""), others.get("chat_w_id", "")
            )

            chat_details = json.loads(response.text)

            old_labels = chat_details.get("labels", [])
            old_labels.extend(["bot_error"])

            print(
                api.assignLabels(
                    others.get("device_id", ""), others.get("chat_w_id", ""), old_labels
                ).text
            )

            if Config.mode != Config.MODE_GOSSIP:
                print(
                    api.assignStatus(
                        others.get("device_id", ""),
                        others.get("chat_w_id", ""),
                        APIWaliChat.STATUS_ACTIVE,
                    ).text
                )

    @staticmethod
    def setBotReplied(APP_CONTEXT: str, others: dict):

        if APP_CONTEXT == "walichat":
            api: APIWaliChat = others.get("api", None)

            if Config.mode != Config.MODE_GOSSIP:
                print(
                    api.assignStatus(
                        others.get("device_id", ""),
                        others.get("chat_w_id", ""),
                        APIWaliChat.STATUS_RESOLVED,
                    ).text
                )


def thread_user_monitor_manage_memory(userMonitor: UserMonitor):

    while True:

        if not userMonitor.alive:
            return
        time.sleep(1)

        id_conversations_to_remove = []

        for brain_name, brain_dict in userMonitor.brain_conversations.items():

            conversations: dict = brain_dict["conversations"]
            
            for unique_id, value in conversations.items():

                conversation: Conversation = value

                if conversation.should_be_deleted:
                    # try to free space
                    while conversation.is_updating_time_to_live:
                        pass

                    conversation.is_updating_time_to_live = True
                    conversation.time_to_live -= 1

                    ttl = conversation.time_to_live

                    # not flood
                    if UserMonitor.verbose > 1:
                        if Config.mode == Config.MODE_TEST:
                            logging.info(
                                f"Conversation {brain_name} {unique_id}: time_to_live {ttl}"
                            )

                    if conversation.time_to_live <= 0:
                        id_conversations_to_remove.append([brain_name, unique_id])

                    conversation.is_updating_time_to_live = False

        for array_remove in id_conversations_to_remove:
            brain_name = array_remove[0]
            unique_id = array_remove[1]

            conversation: Conversation = userMonitor.brain_conversations[brain_name][
                "conversations"
            ][unique_id]

            # doble safe control
            if conversation.should_be_deleted:
                while conversation.is_updating_time_to_live:
                    pass

                if conversation.time_to_live <= 0:

                    length_conversations = len(
                        userMonitor.brain_conversations[brain_name][
                            "conversations"
                        ].items()
                    )
                    logging.info(
                        f"Trying to remove Conversation {brain_name}{unique_id}: actual length: {length_conversations}"
                    )

                    if userMonitor.removeConversationFromMemory(brain_name, unique_id):
                        length_conversations = len(
                            userMonitor.brain_conversations[brain_name][
                                "conversations"
                            ].items()
                        )
                        logging.info(
                            f"Conversation {brain_name} {unique_id}: removed, actual length: {length_conversations}"
                        )
                        if length_conversations == 0:
                            userMonitor.changeShouldBeDeleted(True)
                    else:
                        logging.info(
                            "Conversation %s: Error: cant remove from memory the unique ID",
                            unique_id,
                        )


def test_monitor(
    config: dict,
    brain: BrainRetrieval,
    others: dict,
    conversation: Conversation,
    mongoDatabase: MongoDatabase,
):

    if UserMonitor.verbose > 0:
        logging.info("Thread %s: starting", conversation.unique_id)

    keep_alive = True

    while keep_alive:

        while conversation.counter > 0:

            # not flood
            if UserMonitor.verbose > 1:
                logging.info(
                    f"Thread {conversation.unique_id}: counter {conversation.counter}"
                )

            conversation.counter -= 1
            time.sleep(1)

        while conversation.is_using_message_to_queue:
            pass

        conversation.is_using_message_to_queue = True

        full_text = ""
        for message in conversation.message_queue:
            full_text += message + " "

        original_context = conversation.context

        new_context, msg_list, require_human = brain.thinkAnswer(
            original_context, input_message=full_text, verbose=UserMonitor.brain_verbose
        )

        conversation.context = new_context
        conversation.checkContextInputMemory(brain.chatbots, message)
        conversation.updateRequireHuman(require_human)

        if config["APP_CONTEXT"] == "whatsapp":

            out = ""
            if Config.mode == "test" or Config.mode == "staging":
                out = "*" + conversation.context.getLastTree() + "*"
                out += "\n *" + conversation.context.getLastTags() + "*"

            device_id = conversation.device_id
            chat_w_id = conversation.chat_w_id

            if require_human:

                # print(BrainMonitor.api.assignStatus(device_id, chat_w_id, APIWaliChat.STATUS_PENDING).text)
                # print(BrainMonitor.api.createNote(device_id, chat_w_id, "El bot falló").text)
                # print(BrainMonitor.api.assignLabels(device_id, chat_w_id, ["bot_error"]).text)
                pass
            else:
                # print(BrainMonitor.api.assignStatus(device_id, chat_w_id, APIWaliChat.STATUS_RESOLVED).text)
                pass

            # REMOVE ON PRODUCTION
            if out != "":
                # response = BrainMonitor.api.sendMessage(Monitor.users_monitors[unique_id]["conversation"].mobile, out)
                time.sleep(Config.TIME_SECONDS_WAIT_TEXT_MESSAGE * 2)

            for msg in msg_list:
                response = None
                if "|" in msg:
                    # message with multimedia
                    msg_split = msg.split("|")
                    msg_text = msg_split[0]
                    # id_multimedia = Config.WALI_CHAT_FILES_IDS[msg_split[1]]

                    # response = BrainMonitor.api.sendMessageMultimedia(Monitor.users_monitors[unique_id]["conversation"].mobile, msg_text, id_multimedia)
                    time.sleep(Config.TIME_SECONDS_WAIT_MULTIMEDIA_MESSAGE)
                elif "img" in msg:
                    # id_multimedia = Config.WALI_CHAT_FILES_IDS[msg]
                    # response = BrainMonitor.api.sendMessageMultimedia(Monitor.users_monitors[unique_id]["conversation"].mobile, "", id_multimedia)

                    time.sleep(Config.TIME_SECONDS_WAIT_MULTIMEDIA_MESSAGE)
                else:
                    #  response = BrainMonitor.api.sendMessage(Monitor.users_monitors[unique_id]["conversation"].mobile, msg)
                    time.sleep(Config.TIME_SECONDS_WAIT_TEXT_MESSAGE)
                # logging.info(f"unique id: {unique_id}, sending message response: {response.text}")
                # 1 second delay

        elif config["APP_CONTEXT"] == "telegram":
            # for telegram updater is equal to CallbackContext
            chat_w_id = conversation.chat_w_id

            updater = others["updater"]

            out = "<b><i>" + conversation.context.getLastTree()
            out += "\n" + conversation.context.getLastTags()
            out += "</i></b>"
            updater.bot.sendMessage(chat_id=chat_w_id, parse_mode="HTML", text=out)

            for msg in msg_list:
                updater.bot.sendMessage(
                    chat_id=chat_w_id,
                    parse_mode="HTML",
                    text=msg.replace("<", "").replace(">", ""),
                )

        elif config["APP_CONTEXT"] == "test":
            out = "*" + conversation.context.getLastTree() + "*"
            out += "\n *" + conversation.context.getLastTags() + "*"

            print(out)
            print(msg_list)

        conversation.updateLastBotInteraction()

        count_update = UserMonitor.updateConversation(
            mongoDatabase,
            brain.collection,
            config["APP_CONTEXT"],
            brain.getName(),
            conversation,
        )

        conversation.message_queue = []

        keep_alive = False

        if len(conversation.buffer_message_queue) > 0:
            conversation.message_queue = copy.deepcopy(
                conversation.buffer_message_queue
            )
            # this is important beacause meanwhile we delete someone can send another message
            for buffer_message in conversation.message_queue:
                conversation.buffer_message_queue.remove(buffer_message)

            conversation.counter = Config.TIME_SECONDS_WAIT_BASE_THREAD
            keep_alive = True

        conversation.is_using_message_to_queue = False

    conversation.changeShouldBeDeleted(True)
    if UserMonitor.verbose > 0:
        logging.info("Thread %s: finishing", conversation.unique_id)
