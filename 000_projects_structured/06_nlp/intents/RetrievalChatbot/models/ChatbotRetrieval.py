import random
import json
from typing import Tuple
import numpy as np

import nltk
import copy


import tensorflow as tf
import logging

from tensorflow.python.ops.gen_array_ops import deep_copy

from tensorflow.keras.models import Sequential, load_model as load_model_tensorflow
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics as tf_metrics

from .util.nltk_transforms import *

import os

import math



class ChatbotRetrieval:

    save_path = None
   

    def __init__(self, model_name: str, name: str, context: str, state: str, type: str,
                 intents: dict, nn_config=None, preload_model=False):
        # intents in json like dict
        self.model_name = model_name
        self.name = name
        self.context = context
        self.state = state
        self.type = type
        self.intents = intents
        self.model = None

        self.words = []
        self.classes = []
        self.documents = []

        self.vocabulary_count = {}


        self.directory = os.path.join(ChatbotRetrieval.save_path, model_name, self.name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        if nn_config is not None:
            self.epochs = nn_config['epochs']
            self.batch_size = nn_config['batch_size'] 
            optimizer = nn_config["optimizer"]
            if optimizer["name"] == "Adam":
                self.optimizer = Adam(
                    learning_rate=optimizer["parameters"]["learning_rate"]
                )
            else:
                self.optimizer = Adam()
                print("ERROR", optimizer["name"], "optimizer name not handled")
            self.nn_structure = nn_config["nn_structure"]

        if preload_model:
            self.loadModel()


    @staticmethod
    def convertDocumentToChatbotRetrieval(model_name: str, name: str, document: dict):

        type = document["type"]

        if type == "human":
             return ChatbotRetrieval(model_name, "human", "cb que representa al humano", "", "human", document["tags"])
        else:
            context = document["context"]
            state = document["state"]
            intents = document["intents"]
            nn_config = document["nn_config"]
            
            return ChatbotRetrieval(model_name, name, context, state, type, intents, nn_config=nn_config)

    
    def generateWordsAndClasses(self, MIN_COUNT_WORDS=1, verbose=0):

        for tag, intent in self.intents.items():

            for pattern in intent["training"]:
                # print("PATTERN", pattern)

                word_list, count = clean_count_words(pattern)

                for w in word_list:
                    if w in self.vocabulary_count:
                        self.vocabulary_count[w] += count[w]
                    else:
                        self.vocabulary_count[w] = count[w]


                self.words.extend(word_list)
                self.documents.append((word_list, tag))
                # print("LIST: ", word_list, tag, count)

                if tag not in self.classes:
                    self.classes.append(tag)
                

                
        self.words = sorted(set(self.words))

        self.classes = sorted(set(self.classes))

        if verbose > 0:
            print("CLASSES FOR THIS", self.classes)
        # print(documents)
        # print("WoRDS before", self.words)
        words_copy = copy.deepcopy(self.words)
        self.words = []
        count_removed = 0
        # print("vocabulary_count", self.vocabulary_count)
        for w in words_copy:
            # print(w, self.vocabulary_count[w], MIN_COUNT_WORDS)
            if self.vocabulary_count[w] >= MIN_COUNT_WORDS:
                self.words.append(w)
            else:
                count_removed += 1

        

        if verbose > 0:
            print("\nRemoved", count_removed)
            print("WoRDS clean", self.words)

        self.save_model()
        

    def train(self, porc_train=0.7, porc_val=0.1, verbose=0):
                
        if len(self.classes) == 0:
            self.generateWordsAndClasses(verbose=verbose-1)

        training = []
        output_empty = [0] * len(self.classes)

        # assumption: the words are already processed, for each document
        for document in self.documents:
            bag = []
            # this is the word list
            word_patterns = document[0]
            # word_patterns = [preProcessWord(word.lower()) for word in word_patterns]

            if len(word_patterns) > 0 and verbose > 1:
                print("word_pattern for training", word_patterns)

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            # print(word_patterns)
            # print(bag)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])



        random.shuffle(training)
        training = np.array(training, dtype="object")

        data_x = list(training[:, 0])
        data_y = list(training[:, 1])
        data_length = len(data_x)

        samples_for_train = math.ceil(data_length*porc_train)
        if porc_train == 1:
            samples_for_validation = math.ceil(data_length*porc_train//3)
            samples_for_test = math.ceil(data_length*porc_train//3)

            x_train = np.array(data_x)
            y_train = np.array(data_y)

            index_random_val = list(range(0, data_length))
            index_random_test = list(range(0, data_length))
            random.shuffle(index_random_val)
            random.shuffle(index_random_test)
            index_random_val = index_random_val[0:samples_for_validation]
            index_random_test = index_random_val[0:samples_for_test]

            x_val = x_train[index_random_val]
            y_val = y_train[index_random_val]

            x_test = x_train[index_random_test]
            y_test = y_train[index_random_test]
        
        else:
            samples_for_validation = math.ceil(data_length*porc_val)
            samples_for_test = data_length - samples_for_train - samples_for_validation

            x_train = data_x[0:samples_for_train]
            y_train = data_y[0:samples_for_train]

            x_val = data_x[samples_for_train:samples_for_train+samples_for_validation]
            y_val = data_y[samples_for_train:samples_for_train+samples_for_validation]

            x_test = data_x[samples_for_train+samples_for_validation:data_length]
            y_test = data_y[samples_for_train+samples_for_validation:data_length]

        
        
        
        
        if verbose > 0:
            print(self.model_name, "SAMPLES TO TRAIN: ", data_length, len(x_train), len(x_val), len(x_test))
        # print("TRAIN WITH", x_train)
        # print(samples_for_train, samples_for_validation, samples_for_test)
        if verbose > 0:
            print("x_train[0]", len(x_train[0]))
        # print("x_val[0]", len(x_val[0]))
        # print("y_train[0]", y_train[0])



        input_neurons = len(x_train[0])
        output_neurons = len(y_train[0])

        r = math.floor((input_neurons/output_neurons) ** (1/3))
        h1 = output_neurons * (r ** 2)
        h2 = output_neurons * r

        # print(f"NETWORK BASE CONFIG r={r} [{input_neurons},{h1},{h2},{output_neurons}]")
        # print(f"NETWORK CONFIG r={r} [{input_neurons},{256},{128},{64},{output_neurons}]")

        # best 80 64
        
        self.model = Sequential()
        for layer in self.nn_structure:
            if verbose > 1:
                print("layers", layer)

            if layer["name"] == "input_layer":
                self.model.add(Dense(layer["output_neurons"], input_shape=(input_neurons,), activation=layer["activation_function"]))
            elif layer["name"].lower() == "dropout":
                self.model.add(Dropout(layer["value"]))
            elif layer["name"] == "hidden_layer":
                self.model.add(Dense(layer["output_neurons"], activation=layer["activation_function"]))
            elif layer["name"] == "output_layer":
                self.model.add(Dense(output_neurons, activation=layer["activation_function"]))
            else:
                print("Error, layer malformed", layer)
        
        

        metrics = [
            tf_metrics.CategoricalAccuracy(),
            # tf_metrics.Accuracy(), 
            # tf_metrics.Recall(name="r_recall"), 
            # tf_metrics.Precision(name="r_precision")
            ]

        self.model.compile(
            loss="categorical_crossentropy", 
            optimizer=self.optimizer, 
            metrics=metrics
            )

        fit_verbose = 0
        if verbose in [0,1,2]:
            fit_verbose = verbose

        ## print("FIT VERBOSE", fit_verbose)

        hist = self.model.fit(
            np.array(x_train), np.array(y_train), 
            epochs=self.epochs,  # self.epochs
            batch_size=self.batch_size, # self.batch_size
            verbose=fit_verbose,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(np.array(x_val), np.array(y_val))
            )
        self.model.save(os.path.join(self.directory, "model.h5"), hist)

        
        
        if verbose >= 0:
            print(self.model, len(x_test))
            print(self.name, "TEST EVAL", self.model.evaluate(np.array(x_test), np.array(y_test))) 
        

        return hist.history, self.model.evaluate(np.array(x_test), np.array(y_test))

    def save_model(self):
        with open(os.path.join(self.directory, "words.json"), "w+") as f_words:
            json.dump(self.words, f_words)
            
        with open(os.path.join(self.directory, "classes.json"), "w+") as f_classes:
            json.dump(self.classes, f_classes)

    def loadModel(self):
        if self.name == "human":
            return True
        try:
            self.model = load_model_tensorflow(self.directory + "/model.h5")
            
            with open(os.path.join(self.directory, "words.json"), "r") as f_words:
                self.words = json.load(f_words)
                
            with open(os.path.join(self.directory, "classes.json"), "r") as f_classes:
                self.classes = json.load(f_classes)
            
            return True
        except Exception as exception:
            print("Can't load the model", self.directory, self.name, exception)
            return False

    def predict(self, sentence, ERROR_THRESHOLD=0, MINIMUN_ACCURACY=0.85, verbose=0, reload_model=False) -> Tuple[list,dict]:

        # return the dictionari of the best intent
        if self.model is None or reload_model:
            if not self.loadModel():
                print("CANT PREDICT, MODEL NOT EXISTS ENSURE IS CREATED (TRAINED)")
                raise Exception("CANT PREDICT, MODEL NOT EXISTS ENSURE IS CREATED (TRAINED). If its the first time, even the placeholder brain should be trained")
                return

        
        bow = bag_of_words(sentence, self.words)


        # print(self.classes, self.words, self.model)
        # print("BOW", bow)


        res = self.model.predict(np.array([bow]))[0]


        results = [[i, r] for i, r in enumerate(res)]  #  if r > ERROR_THRESHOLD
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})


        # print("RETURN LIST", return_list)
         

        tag = return_list[0]["intent"]



        if float(return_list[0]["probability"]) < MINIMUN_ACCURACY:
            if verbose > 0:
                print("WARNING: Probability not enought for |", sentence, "|")
            """
            tag = "otros"
            return_list[0]["intent"] = "otrosF"
            """

        # print(len(list_of_intents))

        result_intent = None

        for intent_tag, intent in self.intents.items():
            # print(i["tag"], tag)
            if intent_tag == tag:
                result_intent = intent
                break

        if result_intent is None:
            print("ERROR INTENT", tag, self.name)

        result_intent["tag"] = tag
        
        # print(result)
        return return_list, result_intent

    def evaluateWeakness(self, verbose=0):
        if self.model is None:
            self.loadModel()

        scores = {}


        for tag, intent in self.intents.items():

            scores[tag] = -1
            patterns = intent['training']

            score = 0
            total = 0

            for pattern in patterns:
                prob_list, _ = self.predict(pattern, verbose=(verbose-1))
                prediction_tag = prob_list[0]["intent"]
                
                if tag == prediction_tag:
                    score += 1
                else:
                    if verbose > 0:
                        pass
                        # print("TAG vs PREDICTED", tag, prediction_tag, "pattern", pattern)

                total += 1

            # print("Individual", intent)
            scores[tag] = (score/total) * 100

        return scores

    


