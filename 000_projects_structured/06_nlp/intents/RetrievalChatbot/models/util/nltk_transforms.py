
import nltk
import numpy as np
from nltk.stem import SnowballStemmer

import unicodedata
import re

spanish_stemmer = SnowballStemmer('spanish')

verb_dict = {
    'q': 'que',
    'x': 'por',
    'd': 'de',
    'k': 'que',
    'atravez': 'a traves',
    'tb': 'tambien',
    'haceerle': 'hacerle',
    'compañia': 'compania',
    'aser': "hacer",
    'veder': "vender",
    "rub": "ruc",
    "fatura": "factura",
    "eletronica": "electronica",
    "haorita": "ahorita",
    "presio": "precio",
    "realizdo": "realizado",
    "natura": "natural",
    "info": "informacion",
    "clmo": "como",
    "cimo": "como",
    "mail": "correo",
    "email": "correo",
    "eata": "esta",
    "realizdo": "realizado",
    "cualqueir": "cualquier",
    "ola": "hola"

}

def bag_of_words(sentence, words):
    
    sentence_words, count = clean_count_words(sentence)
    bag = [0] * len(words)
    
   #  print("TO PREDICT: bag_of_words: ", sentence_words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)




def unicodeToAscii(s):
    
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    
    s = unicodeToAscii(s.lower().strip())
    # print(s)
    # TO-DO validate if is url not separate dot
    s = re.sub(r"\s+", r" ", s).strip()

    new_s = ""
    s_split = s.split(" ")
    for word in s_split:
        t_word = word
        if "http" not in t_word:

            for key, value in verb_dict.items():
                if t_word == key:
                    t_word = value

            # replace special
            t_word = re.sub(r"([,.!?])", r" \1 ", t_word)
            # print("1|%s|" % t_word)
            t_word = re.sub(r"[^üáéíóúa-zA-Z\d]+", r" ", t_word)
            # print("2|%s|" % t_word)
            t_word = re.sub(r"\s+", r" ", t_word).strip()
            # print("3|%s|" % t_word)
            t_word = preProcessWord(t_word)

            


        new_s += t_word + " "

    s = re.sub(r"\s+", r" ", new_s).strip()
    s = s.replace("á", "a")
    s = s.replace("é", "e")
    s = s.replace("í", "i")
    s = s.replace("ó", "o")
    s = s.replace("ú", "u")
    s = s.replace("ü", "u")
    return s

def preProcessWord(word: str)->str:
    new_word = word
    try:
        new_word = spanish_stemmer.stem(new_word)
    except Exception as error:
        print("Error para palabra", word)
    return new_word

def clean_count_words(text):
    ignore_letters = [
        ".", "?", "¡", "!", "¿", ",", "de", "en", "la", "el",
        "que", "un", "una", "etc", "amigo", "y", "pregunta",
        "duda", "a", "vi", "con", "i", "e", "o", "u",
        ]

    words = nltk.word_tokenize(normalizeString(text))

    clean = []
    count = {}

    for word in words:
        new_word = word
        
      
        

        if new_word not in ignore_letters:
            clean.append(new_word)

            if new_word in count:
                count[new_word] += 1
            else:
                count[new_word] = 1

    return list(set(clean)), count