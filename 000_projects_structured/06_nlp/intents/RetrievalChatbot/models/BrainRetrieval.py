
from .ChatbotRetrieval import ChatbotRetrieval
from .Conversation import *
import re
import copy
import random
from .Config import Config

class BrainRetrieval:


    def __init__(self, model_name: str, chatbots: dict, customManageMemory, customManageState, 
       customManageContextMessage, customManageHuman,  collection: str="conversations"):
        self.chatbots = {}

        self.name = model_name
        self.collection = collection

        self.customManageMemory = customManageMemory
        self.customManageState = customManageState
        self.customManageContextMessage = customManageContextMessage
        self.customManageHuman = customManageHuman


        ChatbotRetrieval.save_path = Config.models_saved

        for _id, document_chatbot in chatbots.items():

            self.chatbots[_id] = ChatbotRetrieval.convertDocumentToChatbotRetrieval(model_name, _id, document_chatbot)


    def getName(self):
        return self.name
    
    def reloadModels(self):
        for chatbot_id, chatbot in self.chatbots.items():
            chatbot.loadModel()

    
    def manageMemory(self, context: ConversationContext, best_intent: dict, verbose=0) -> ConversationContext:

        newContext = self.customManageMemory(context, best_intent, verbose)
        

        # other stuff but for ALL clients

        return newContext

    def manageState(self, context: ConversationContext, best_intent: dict, verbose=0) -> ConversationContext:

        newContext = self.customManageState(context, best_intent, verbose)
        
        print("MANAGE STATE", best_intent)
        if "state" in best_intent:
            # the intent is a posible state
            new_state = best_intent["state"]
            print("IS VALID", new_state, BrainRetrieval.isValidState(new_state))

            if BrainRetrieval.isValidState(new_state):
                newContext.addState(new_state)


        return newContext

    @staticmethod
    def isValidState(state):
        return state != "null" and state != "" and state != None


    def getMessage(self, context: ConversationContext, best_intent: dict, verbose=0):

        # print("DEbug best intent", best_intent["responses"])
        formated_response: list = best_intent["responses"]
        route_to_no_context = False

        # if require_context then the first element should be the filter
        if best_intent.get("require_context", False):

            formated_response, route_to_no_context = self.customManageContextMessage(context, best_intent, formated_response, route_to_no_context, verbose)



        return formated_response, route_to_no_context


    def getNextChatbotName(self, context: ConversationContext, best_intent: dict, route_to_no_context: bool, verbose=0) -> str:

        next_chatbot_name = best_intent["next_chatbot_name"]

        if route_to_no_context:
            next_chatbot_name = best_intent["next_chatbot_name_if_no_context"]

        return next_chatbot_name


    
            
    def getBroker(self):
        broker: ChatbotRetrieval = None
        first = True
        for chatbot_id, chatbot_object in self.chatbots.items():
            if first:
                first_chatbot = chatbot_object
                first = False
            if chatbot_object.type == "broker":
                broker = chatbot_object

        if broker is None:
            broker = first_chatbot

        return broker


    def predictRequest(self, context: ConversationContext, message: str, verbose=0):


        chatbot: ChatbotRetrieval = self.chatbots.get(context.current_agent, self.getBroker())
        probabilities, best_intent = chatbot.predict(message)
        if verbose > 0:
            print("best_intent probabilities", probabilities)

        if verbose > 0:
            print("**************BRAIN START**************")
            print("*Context initial", context)
        context = self.manageMemory(context, best_intent, verbose)
        if verbose > 0:
            print("*Context after manage memory", context)
        # manage intent state
        context = self.manageState(context, best_intent, verbose)
        if verbose > 0:
            print("*Context after manage state", context)

        formated_msg = []
        route_to_no_context = False
        if best_intent["is_natural_language_tag"]:
            formated_msg, route_to_no_context = self.getMessage(context, best_intent, verbose)
            
        # print("*formated_msg", formated_msg)
        
        next_chatbot_name = self.getNextChatbotName(context, best_intent, route_to_no_context, verbose)
        if verbose > 0:
            print("*next_chatbot_name", next_chatbot_name)
            
        if next_chatbot_name in self.chatbots and next_chatbot_name != "human":

            # this chatbot determinate a state
            new_state = self.chatbots[next_chatbot_name].state
            print("New general state", new_state, BrainRetrieval.isValidState(new_state))
            if BrainRetrieval.isValidState(new_state):
                context.addState(new_state)
                print("Context states between", context.states)
               
        else:
            if verbose > 0:
                print("next_chatbot_name not in chatbots", next_chatbot_name)


        
        if verbose > 0:
            print("**************BRAIN END**************")

        return probabilities, formated_msg, next_chatbot_name


    def thinkAnswer(self, original_context: ConversationContext, input_message: str, MAX_NEURAL_NETWORK_LOOP=10, verbose=0):
        # return new_context, msg, request_human



        counter = 0
        request_human = False
        new_context = original_context # copy.deepcopy(original_context)  # a copy for safe

        last_agents = []
        last_tags = []

        is_requesting_data = False
        

        all_tags_probabilities = []
        while counter < MAX_NEURAL_NETWORK_LOOP:

            
                
            last_agents.append(new_context.current_agent)

            print("States before", new_context.states)
            tags_probabilities, msg_list, next_agent = self.predictRequest(new_context, input_message, verbose)
            
            print("States after", new_context.states)

            proccessed_tags_probabilities = dict()
            for tag_probability in tags_probabilities:
                proccessed_tags_probabilities[tag_probability['intent']] = round(float(tag_probability['probability']) * 100, 4)

            all_tags_probabilities.append(proccessed_tags_probabilities)
            

            last_tags.append(tags_probabilities[0]['intent'])


            if next_agent == "human":
                new_context.current_agent = "broker"
                request_human = True
            elif next_agent == "last":
                new_context.current_agent = last_agents[-2]
            else:
                new_context.current_agent = next_agent



            counter += 1

            if len(msg_list) > 0 or request_human:
                # message delivered, so end  or human need              
                break
            
            elif counter == MAX_NEURAL_NETWORK_LOOP or (next_agent in last_agents and next_agent not in ["last"]):
                new_context.current_agent = "broker"
                request_human = True
                break


        last_agents.append(new_context.current_agent)
        all_tags_probabilities.append(dict())
        # last_agents.append(new_context.current_agent)



        if request_human:

            msg_human = self.customManageHuman(self.chatbots["human"], new_context, input_message, last_agents, last_tags)


            msg_list.extend(msg_human)


        
        
        new_context.addHistory(input_message, last_agents, last_tags, msg_list, request_human, all_tags_probabilities)

        return new_context, msg_list, request_human
        


