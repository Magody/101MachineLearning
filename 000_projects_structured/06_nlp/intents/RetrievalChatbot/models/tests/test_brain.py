from ..BrainRetrieval import BrainRetrieval
from ..ChatbotRetrieval import ChatbotRetrieval

from ..util.data_preparation import *

from ..Conversation import *
from ..util.mongo_util import *
from ..custom_bussiness import customManageMemory, customManageState, customManageContextMessage, customManageHuman


def test_brain(collection_name_conversations, mongoDatabase: MongoDatabase, brain_name, chatbots):

    brain = BrainRetrieval(brain_name, chatbots, customManageMemory, customManageState, customManageContextMessage, customManageHuman)


    conversations = {}



    while True:

        conversation_id = "test"
        message = input(">>you (send q for quit): ")
        if message == "q":
            break

        if conversation_id not in conversations:
            # first search in db to try to recover
            conversation = mongoDatabase.find(collection_name_conversations, {"_id": conversation_id})

            if conversation:
                print("Search in db")
                conversations[conversation_id] = Conversation.convertMongoDBDocumentToObject(conversation)
            else:
                # save new phone
                conversations[conversation_id] = Conversation(conversation_id, conversation_id, ConversationContext("broker"), "n/a", "n/a", message=message)
                id = mongoDatabase.insertDocument(collection_name_conversations, conversations[conversation_id].getDocumentForMongoDB())
                print("Inserted id", id)

        conversation: Conversation = conversations[conversation_id]
                
        conversation.message = message
        
        original_context = conversation.context

        new_context, msg_list, require_human = brain.thinkAnswer(original_context, input_message=message, verbose=2)

        conversation.context = new_context
        conversation.checkContextInputMemory(brain.chatbots, message)
        conversation.updateRequireHuman(require_human)


        full_text = ""
        for msg in msg_list:
            full_text += msg + "->"

        out = "" + conversation.context.getLastTree()
        out += "\n" + conversation.context.getLastTags()
        out += "\n"
        print(out, full_text)

        modified_count = mongoDatabase.update(collection_name_conversations, {"_id": conversation_id}, conversation.getDocumentForMongoDB())
        print("Modified count", modified_count)

