
from datetime import datetime
import pytz
import json
from collections import OrderedDict
from .Config import Config
from .util.TimeUtil import TimeUtil
from .ChatbotRetrieval import ChatbotRetrieval
import math

class ConversationContext:

    def __init__(self, current_agent: str):
        self.current_agent = current_agent
        self.history = OrderedDict()


        self.tags_first_message_sent =[]
        self.states = OrderedDict()
        self.memory_intents = []
        self.person_type = ""
        self.input_memory = OrderedDict()

    def __str__(self) -> str:
        return f"current_agent:{self.current_agent} | tags_first_message_sent:{self.tags_first_message_sent} | states:{self.states} | memory_intents:{self.memory_intents} | person_type:{self.person_type}"

    def addHistory(self, message: str, agents: list, tags: list, bot_message: list, 
            require_human: bool, tags_probabilities: list = []):
        
        
        date_time_string = TimeUtil.getNowStringWithTimeZone()
        self.history[date_time_string] = dict()
        
        self.history[date_time_string]["client_message"] = message
        self.history[date_time_string]["bot_message"] = bot_message
        self.history[date_time_string]["agents"] = agents
        self.history[date_time_string]["tags"] = tags
        self.history[date_time_string]["require_human"] = require_human

        

        self.history[date_time_string]["tags_probabilities"] = tags_probabilities


    

    def addInputMemory(self, chatbot_name: str, message: str):
        
        
        date_time_string = TimeUtil.getNowStringWithTimeZone()
        self.input_memory[date_time_string] = dict()
        
        self.input_memory[date_time_string]["chatbot_name"] = chatbot_name
        self.input_memory[date_time_string]["message"] = message

    def getLastTree(self) -> str:
        out = "(Tree)"
        last_key = next(reversed(self.history))
        last_data = self.history[last_key]

        for index, agent in enumerate(last_data["agents"]):
            out += agent
            if index < len(last_data["agents"]) - 1:
                out += "->"
            else:
                out += "."

        return out

    def getLastTags(self) -> str:
        out = "(Tags)"

        last_key = next(reversed(self.history))
        last_data = self.history[last_key]

        for index, tag in enumerate(last_data["tags"]):
            out += tag
            if index < len(last_data["tags"]) - 1:
                out += "->"
            else:
                out += "."

        return out

    def getLastProbabilities(self) -> str:
        out = "(LastProbabilities)"

        last_key = next(reversed(self.history))
        last_data = self.history[last_key]

        for index, last_tag_probability in enumerate(last_data["tags_probabilities"]):
            # print(last_data)
            out += json.dumps(last_tag_probability, separators=(',', ':'))
            if index < len(last_data["tags_probabilities"]) - 1:
                out += "-->"
            else:
                out += "."

        return out



    def addState(self, new_state: str):
        print("Conversation: addState: Adding new state", new_state)
        date_time_string = TimeUtil.getNowStringWithTimeZone()        
        self.states[date_time_string] = new_state


    def getDocumentForMongoDB(self) -> dict:


        return {
            "current_agent": self.current_agent,
            "history": self.history,
            "tags_first_message_sent": self.tags_first_message_sent,
            "states": self.states,
            "memory_intents": self.memory_intents,
            "person_type": self.person_type,
            "input_memory": self.input_memory
        }

    @staticmethod
    def convertMongoDBDocumentToObject(document: dict):
        context = ConversationContext(document["current_agent"])
        context.history = OrderedDict(document["history"])
        context.tags_first_message_sent = document["tags_first_message_sent"]
        context.states = OrderedDict(document["states"])
        context.memory_intents = document["memory_intents"]
        context.person_type = document["person_type"]
        context.input_memory = OrderedDict(document.get("input_memory", dict()))
        return context

        




class Conversation:

    def __init__(self, 
                unique_id: str, mobile: str, context: ConversationContext,
                device_id: str, chat_w_id: str,
                message: str = "", name: str = "", require_human: bool = None,
                date_require_human: str = "", 
                date_last_client_interaction: str = "",
                date_last_bot_interaction: str = "", username: str = "", app_context: str = "",
                brain_name: str = "", date_first_client_interaction: str = "",
                show_bot_presentation: bool = True
                ):
        self.unique_id = unique_id
        self.mobile = mobile
        self.context = context
        self.device_id = device_id
        self.chat_w_id = chat_w_id
        self.message = message
        self.name = name
        if require_human is None:
            self.require_human = False
        else:
            self.require_human = require_human
        
        
        self.date_require_human = date_require_human
        self.date_last_client_interaction = date_last_client_interaction
        self.date_last_bot_interaction = date_last_bot_interaction
        self.username = username
        self.app_context = app_context
        self.brain_name = brain_name
        self.date_first_client_interaction = date_first_client_interaction
        self.show_bot_presentation = show_bot_presentation

        # control variables conversation
        
        self.counter = Config.TIME_SECONDS_WAIT_BASE_THREAD
        self.is_creating_thread = False
        self.is_thread_created = False
        self.is_using_message_to_queue = False
        self.thread = None
        self.message_queue = []
        self.buffer_message_queue = []
        self.is_being_deleted = False
        self.should_be_deleted = False
        self.resetTimeToLive()

    def checkContextInputMemory(self, chatbots: dict, message: str):
        last_key = next(reversed(self.context.history))
        last_data = self.context.history[last_key]
        agent_before_current: str = last_data["agents"][-2]

        last_chatbot: ChatbotRetrieval = chatbots[agent_before_current]
        # LAST CHATBOT", agent_before_current, "type", last_chatbot.type)

        if last_chatbot.type == "input_collector":
            if "cb_ask_name" == agent_before_current:
                self.name = message
            else:
                self.context.addInputMemory(agent_before_current, message)

    def resetTimeToLive(self):
        self.is_updating_time_to_live = True
        self.time_to_live = Config.TIME_SECONDS_WAIT_FOR_FREE_MEMORY
        self.is_updating_time_to_live = False

    def resetContext(self):
        self.require_human = False
        self.date_require_human = ""
        self.date_last_client_interaction = ""
        self.date_last_bot_interaction = ""
        self.context = ConversationContext("broker")

    def changeShouldBeDeleted(self, condition: bool):
        self.should_be_deleted = condition
        if not condition:
            self.resetTimeToLive()


    def __str__(self) -> str:
        return f"mobile:{self.mobile}|[context:{self.context}]|message:{self.message}|"

    def updateRequireHuman(self, new_value: bool):
        if new_value:
            # pass from False to True
            date_time_string = TimeUtil.getNowStringWithTimeZone()
            self.date_require_human = date_time_string
            self.show_bot_presentation = True

        self.require_human = new_value

        
    def updateFirstClientInteraction(self, date_time_string=""):
         # first time pass from False to True
        if date_time_string != "":
            self.date_first_client_interaction = date_time_string
        else:     
            self.date_first_client_interaction = TimeUtil.getNowStringWithTimeZone()  


    def updateLastClientInteraction(self):
         # first time pass from False to True
        date_time_string = TimeUtil.getNowStringWithTimeZone()       
        self.date_last_client_interaction = date_time_string

    def updateLastBotInteraction(self):
        
        date_time_string = TimeUtil.getNowStringWithTimeZone()    
        self.date_last_bot_interaction = date_time_string


    def getDocumentForMongoDB(self) -> dict:
        
        return {
            "_id": self.unique_id,
            "mobile": self.mobile,
            "device_id": self.device_id,
            "chat_w_id": self.chat_w_id,
            "name": self.name,
            "require_human": self.require_human,
            "last_message": self.message,
            "date_require_human": self.date_require_human,
            "date_first_client_interaction": self.date_first_client_interaction,
            "date_last_client_interaction": self.date_last_client_interaction,
            "date_last_bot_interaction": self.date_last_bot_interaction,
            "context": self.context.getDocumentForMongoDB(),
            "username": self.username,
            "app_context": self.app_context,
            "brain_name": self.brain_name,
            "show_bot_presentation": self.show_bot_presentation
        }

    @staticmethod
    def convertMongoDBDocumentToObject(document: dict):
        
        
        return Conversation(
            document['_id'], 
            document['mobile'], 
            ConversationContext.convertMongoDBDocumentToObject(document['context']),
            document['device_id'], 
            document['chat_w_id'],  
            message=document['last_message'],
            name=document["name"],
            require_human=document["require_human"],
            date_require_human=document["date_require_human"],
            date_last_client_interaction=document["date_last_client_interaction"],
            date_last_bot_interaction=document["date_last_bot_interaction"],
            username=document["username"],
            app_context=document["app_context"],
            brain_name=document["brain_name"],
            date_first_client_interaction=document.get("date_first_client_interaction", None),
            show_bot_presentation=document.get("show_bot_presentation", True)
            )

    def allowCommunicationEvenHumanRequest(self):
        
        datetime_last_human_request = datetime.strptime(self.date_require_human, "%Y-%m-%d %H:%M:%S")
            
        datetime_last_human_request_with_timezone = datetime_last_human_request
        now_with_timezone = TimeUtil.getNowWithTimeZone()

        (hours, minutes, _) = TimeUtil.getDifferenceBetweenDatetimes(now_with_timezone, datetime_last_human_request_with_timezone)

        total_minutes = (hours * 60) + minutes

        # print("Conversation: allowCommunicationEvenHumanRequest: total_minutes", total_minutes, Config.TIME_MINUTES_BOT_WAIT_TO_RESET)

        return total_minutes >= Config.TIME_MINUTES_BOT_WAIT_TO_RESET

    




