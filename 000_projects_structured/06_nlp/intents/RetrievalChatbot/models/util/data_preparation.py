import os
import json
import math
import time

from models.db.MongoDatabase import MongoDatabase

from ..Config import Config

def convert_list_to_string(array: list) -> str:
    out = ""
    for index, element in enumerate(array):
        out += element.replace("\n","")
        if index < len(array)-1:
            out += "->"

    return out

def searchBranch(level: int, search: str, tree: dict):

    if len(tree) == 0 or tree is None:
        return


    for key_branch, branch in tree.items():

        if key_branch == search:
            return branch
        
        searchBranch(level+1, key_branch, branch)

def getChatbotsWithTagsWithResponse(path_folder_brain_flow):

    folders = list(filter(lambda filename: "txt" not in filename, os.listdir(path_folder_brain_flow)))
    chatbots_list = []
    for folder in folders:


        f_intents = open(os.path.join(path_folder_brain_flow, folder, "info.json"), "r")
        intents = json.load(f_intents)["intents"]

        tags = []
        for intent in intents:
            if intent['is_natural_language_tag']:
                tags.append(intent['tag'])


        f_intents.close()

        if len(tags) > 0:
            chatbots_list.append([folder, tags])
   
    return chatbots_list


def readBranchsRecursively(path_brain_flow, tree):

    if len(tree) == 0 or tree is None:
        return []

    branchs = {}

    for key_branch, branch in tree.items():

        if key_branch not in branchs:
            branchs[key_branch] = {}

        path_folder_chatbots = os.path.join(path_brain_flow, key_branch)

        txts = list(filter(lambda filename: "txt" in filename, os.listdir(path_folder_chatbots)))
    
        if len(txts) > 0:
            branchs[key_branch]["txts"] = txts

        childs = readBranchsRecursively(path_brain_flow, branch)
        if len(childs) > 0:
            branchs[key_branch]["branchs"] = childs

    return branchs


# deprecated
def loadAgentsInfo(agents_base, chatbots_path, verbose=0, BASE_OTHERS=40):
    agents = agents_base


    for key, value in agents.items():

        agent_path = os.path.join(chatbots_path, 'brain_flow', key)

        with open(os.path.join(agent_path, 'info.json'), encoding='utf-8') as file_info:
            
            intents = json.load(file_info)
            count_patterns = 0
            for intent in intents['intents']:
                patterns = []
                tag = intent['tag']

                


                if tag == "otros":
                    for _ in range(BASE_OTHERS + count_patterns):  # count_patterns
                        patterns.append("")
                    intent["patterns"] = patterns
                    # no count, is not useful for augmentation? count_patterns += len(intent["patterns"])
                    continue


                patterns_path = agent_path
                if not intent['is_natural_language_tag'] and (tag.startswith("cb") or tag == "my_broker"):
                    patterns_path = os.path.join(chatbots_path, 'brain_flow')

                    if tag == "my_broker":
                        # this means, all other data but not data on my forward branch
                        f_broker = open(os.path.join(patterns_path, f"broker.txt"), "r", encoding="utf-8")

                        f_branch = open(os.path.join(patterns_path, f"{key}.txt"), "r", encoding="utf-8")

                        final_output = []

                        lines_branch = f_branch.readlines()
                        # print("lines_branch", lines_branch)
                        for line in f_broker.readlines():
                            if line not in lines_branch:
                                final_output.append(line)


                        
                        patterns = list(map(lambda string: string.replace("\n", ""), final_output)) 
                        """
                        f_dump = open(os.path.join(patterns_path, f"no{key}.txt"), "w", encoding="utf-8")
                        f_dump.writelines(final_output)
                        f_dump.close()
                        """

                        f_broker.close()
                        f_branch.close()

                        # print("PATTERNS GET DEB", patterns)
                    else:

                        # save patterns from TXT
                        with open(os.path.join(patterns_path, tag + '.txt'), encoding='utf-8') as file_txt:
                            patterns = list(map(lambda string: string.replace("\n", ""), file_txt.readlines()))

                else:
                     # save patterns from TXT
                    with open(os.path.join(patterns_path, tag + '.txt'), encoding='utf-8') as file_txt:
                        patterns = list(map(lambda string: string.replace("\n", ""), file_txt.readlines()))

                intent["patterns"] = patterns
                count_patterns += len(patterns)

            mean_patterns = math.ceil(count_patterns/len(intents['intents']))

            
            for intent in intents['intents']:
                actual_length = len(intent["patterns"])
                if actual_length < mean_patterns:
                    multiply_length = math.ceil(mean_patterns/actual_length)
                    base_copy = intent["patterns"]
                    intent["patterns"] = []
                    for _ in range(multiply_length):
                        intent["patterns"].extend(base_copy)

                    if verbose > 0:
                        print("OVERFITING WARNING, aumentar más datos:", key, intent["tag"], "MEAN", mean_patterns, "Actual length for intent", actual_length)
                    
                        print("BEFORE", actual_length, "NOW", len(intent["patterns"]))

            
           
            
            agents[key] = intents
            # print("\n\n", key, "\n\n", intents)
            
    return agents


def fill_non_natural_language_intents(path_brain_flow: str, documents_chatbots: dict, verbose=0, BASE_OTHERS=40)->dict:
    """Inflates/Fill training data for common intents/redirects. Also handles the imbalanced data,
    trying to set equal number of training data for each tag.
    
    - For intent of type "otros" will create empty values as training.
    - For intents of type "cb*" (no natural language only redirect) it collects training data generated
        for that specific chatbot. Example: cb_signature will have a file cb_signature.txt
        with all phrases from all its intents.
    - For intents of type "my_broker" (redirect to broker). It collects all intents in the systems minus the intents
        in the current branch. This is, predicts every topic different to actual. So, will probably
        return to broker.
    

    Args:
        path_brain_flow (str): folder path where the TXT are being generated with intents.
        documents_chatbots (dict): Chatbots information of brain. This dict should have training info. 
            It will get a copy and return the copy with modifications
        verbose (int, optional): Verbose level for log messages. Defaults to 0.
        BASE_OTHERS (int, optional): Number of empty samples to fill in this intent. Defaults to 40.

    Returns:
        dict: document_chatbot with the training info augmented for every no natural language intent.
    """
    
    
    agents = documents_chatbots
    # print("BEGIN", agents)
    for key, document_chatbot in agents.items():

        if(key == "human"):
            continue

        intents: dict = document_chatbot["intents"]

        count_patterns = 0
        for tag, intent in intents.items():
            training_patterns = []

            if tag == "otros":
                for _ in range(BASE_OTHERS + count_patterns):  # count_patterns
                    training_patterns.append("")
                intent["training"] = training_patterns
                # no count, is not useful for augmentation? count_patterns += len(intent["patterns"])
                continue

            if not intent.get("is_natural_language_tag", False) and (tag.startswith("cb") or tag == "my_broker"):
                
                # Collect broker training data and eliminates branch training data
                if tag == "my_broker":
                    # this means, all other data but not data on my forward branch
                    f_broker = open(os.path.join(path_brain_flow, f"broker.txt"), "r", encoding="utf-8")

                    f_branch = open(os.path.join(path_brain_flow, f"{key}.txt"), "r", encoding="utf-8")

                    final_output = []

                    lines_branch = f_branch.readlines()
                    # print("lines_branch", lines_branch)
                    for line in f_broker.readlines():
                        if line not in lines_branch:
                            final_output.append(line)


                    
                    training_patterns = list(map(lambda string: string.replace("\n", ""), final_output)) 
                    """
                    f_dump = open(os.path.join(patterns_path, f"no{key}.txt"), "w", encoding="utf-8")
                    f_dump.writelines(final_output)
                    f_dump.close()
                    """

                    f_broker.close()
                    f_branch.close()

                    # print("PATTERNS GET DEB", patterns)
                else:

                    # save patterns from TXT
                    with open(os.path.join(path_brain_flow, tag + '.txt'), encoding='utf-8') as file_txt:
                        training_patterns = list(map(lambda string: string.replace("\n", ""), file_txt.readlines()))

           

                intent["training"] = training_patterns
                count_patterns += len(training_patterns)

        mean_patterns = math.ceil(count_patterns/len(intents))

        
        for tag, intent in intents.items():
            # print("DEBUG", key, tag, intent)
            
            actual_length = len(intent["training"])

            if actual_length < mean_patterns:
                multiply_length = math.ceil(mean_patterns/actual_length)
                base_copy = intent["training"]
                intent["training"] = []
                for _ in range(multiply_length):
                    intent["training"].extend(base_copy)

                if verbose > 0:
                    print("OVERFITING WARNING, aumentar más datos:", key, tag, "MEAN", mean_patterns, "Actual length for intent", actual_length)
                
                    print("BEFORE", actual_length, "NOW", len(intent["training"]))

        
        
        
        document_chatbot["intents"] = intents

    return agents



def concatBranch(level: int, father: str, tree: dict, path_brain_flow: str):

    if len(tree) == 0 or tree is None:
        return father

    out = []
    for key, branch in tree.items():
        print("-"*level, father, key)
        out.append(concatBranch(level + 1, key, branch, path_brain_flow))

    print("Father", father, "contains", out)

    lines_out = ""
    for folder in out:

        folder_path = os.path.join(path_brain_flow, folder)



        # txt inside folder, to append to previous data
        txt_files = list(filter(lambda filename: "txt" in filename, os.listdir(folder_path)))
        
        out_cb = ""
        for txt in txt_files:
            f = open(os.path.join(folder_path, txt), "r", encoding="utf-8")
            file_text = f.read()

            if not file_text.endswith("\n"):
                file_text += "\n"

                
            out_cb += file_text

            f.close()

        


        
        
        f = open(os.path.join(path_brain_flow, f"{folder}.txt"), "a+", encoding="utf-8")
        f.write(out_cb)
        f.close()

        # after append data existing in folder, get all this with last appends from childs

        f_complete_read = open(os.path.join(path_brain_flow, f"{folder}.txt"), "r", encoding="utf-8")
        lines_out += f_complete_read.read()
        f_complete_read.close()


        print(txt_files, "with data recovered by", folder)

    f = open(os.path.join(path_brain_flow, f"{father}.txt"), "a+", encoding="utf-8")
    f.write(lines_out)
    f.close()



    return father

def generateConcats(path_brain_flow):
    files_txt = list(filter(lambda filename: "txt" in filename, os.listdir(path_brain_flow)))
    for file in files_txt:
        os.remove(os.path.join(path_brain_flow, file))
    concatBranch(1, "broker", Config.tree_broker, path_brain_flow)


def concatBranchFromArrays(level: int, father: str, tree: dict, path_brain_flow: str, documents_chatbot: dict, verbose=0):

    if len(tree) == 0 or tree is None:
        return father

    out = []
    for key, branch in tree.items():
        if verbose > 0:
            print("-"*level, father, key)
        out.append(concatBranchFromArrays(level + 1, key, branch, path_brain_flow, documents_chatbot))

    if verbose > 0:
        print("Father", father, "contains", out)

    lines_out = ""
    for folder in out:

        tags = documents_chatbot[folder]["intents"]

        out_cb = ""
        for tag, intent in tags.items():
            # print("DEBUG", tag)

            if "training" in intent:
                out = ""
                for training_sample in intent["training"]:
                    out += training_sample + "\n"
                out_cb += out
        
        
        f = open(os.path.join(path_brain_flow, f"{folder}.txt"), "a+", encoding="utf-8")
        f.write(out_cb)
        f.close()

        # after append data existing in folder, get all this with last appends from childs

        f_complete_read = open(os.path.join(path_brain_flow, f"{folder}.txt"), "r", encoding="utf-8")
        lines_out += f_complete_read.read()
        f_complete_read.close()


        if verbose > 0:
            print("data recovered by", folder)

    f = open(os.path.join(path_brain_flow, f"{father}.txt"), "a+", encoding="utf-8")
    f.write(lines_out)
    f.close()



    return father

def generateConcatsFromArrays(path_brain_flow, tree_broker, documents_chatbot, verbose=0):
    files_txt = list(filter(lambda filename: "txt" in filename, os.listdir(path_brain_flow)))
    for file in files_txt:
        os.remove(os.path.join(path_brain_flow, file))

    concatBranchFromArrays(1, "broker", tree_broker, path_brain_flow, documents_chatbot, verbose)

def getChatbotsNamesInDict(path_folder_chatbots):
    folders = list(filter(lambda filename: "txt" not in filename, os.listdir(path_folder_chatbots)))
    chatbots_dict = dict()
    for folder in folders:
        chatbots_dict[folder] = {}
   
    return chatbots_dict