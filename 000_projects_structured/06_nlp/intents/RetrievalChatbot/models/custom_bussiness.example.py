
from .ChatbotRetrieval import ChatbotRetrieval
from .BrainRetrieval import BrainRetrieval
from .Conversation import ConversationContext
import copy
from .util.TimeUtil import TimeUtil

"""
Principal functions
"""

def customManageMemory(context: ConversationContext, best_intent: dict, verbose=0) -> ConversationContext:
    newContext = context  # always pass by reference
    if "remember" in best_intent:
        # this intent should not be forget
        tag = best_intent["tag"]
        if "info_natural" == tag or "info_juridica" == tag or "info_servidor_publico" == tag:
            context.person_type = tag
        else:
            context.memory_intents.append(tag)

    return newContext
    
def customManageState(context: ConversationContext, best_intent: dict, verbose=0) -> ConversationContext:
    newContext = context  # always pass by reference
    
    
            
    return newContext

def customManageContextMessage(context: ConversationContext, best_intent: dict, formated_response, route_to_no_context, verbose=0):
    
    new_formated_response = copy.deepcopy(formated_response)
    new_route_to_no_context = copy.deepcopy(route_to_no_context)
    # the format is [context_key_code,msg], only exist one response
    context_key_code = new_formated_response[0]

    if "person" in context_key_code:



        if context.person_type != "":
            extract_specific = None
            try:
                extract_specific = best_intent["context_responses"].get(context_key_code+"_"+context.person_type, None)
            except Exception as error:
                pass
            
            if extract_specific is None:
                new_route_to_no_context = True
                extract_specific = new_formated_response[1]
            
            
            new_formated_response = [extract_specific]
        else:
            new_route_to_no_context = True
            new_formated_response.remove(context_key_code)
    else:
        route_to_no_context = True
        formated_response.remove(context_key_code)

    return new_formated_response, new_route_to_no_context 

""" 
elif "contexto" in context_key_code:
        key_context = ""
        print("Debug states", context.states)
        for key, value in context.states.items():
            print(key, value)
            if "direccion_recolectada" in value:
                key_context = "contexto1"
                break

        print("key", key_context)
        if key_context != "":
            extract_specific = None
            try:
                print("context responses", best_intent["context_responses"])
                extract_specific = best_intent["context_responses"].get(key_context, None)
            except Exception as error:
                pass
            print("extract especific", extract_specific)
            if extract_specific is None:
                new_route_to_no_context = True
                extract_specific = new_formated_response[1]


            new_formated_response = [extract_specific]
            print("final response", new_formated_response)
        else:
            route_to_no_context = True
            formated_response.remove(context_key_code)
"""


def customManageHuman(chatbot_human: ChatbotRetrieval, new_context, input_message, last_agents, last_tags):

    is_laboral_time = TimeUtil.isLaboralTime(0)
    

    if is_laboral_time:
        msg_human = chatbot_human.intents["message_human_in_laboral_time"]["responses"]
       
    else:
        msg_human = chatbot_human.intents["message_human_out_laboral_time"]["responses"]

    for agent in last_agents:
        if "factuexpert" in agent:
            if is_laboral_time:
                msg_human = chatbot_human.intents["message_human_just_routing_in_laboral_time"]["responses"]
            else:
                msg_human = chatbot_human.intents["message_human_just_routing_out_laboral_time"]["responses"]

    return msg_human

"""
Secundary functions
"""


