import pandas as pd
import re
from bs4 import BeautifulSoup
import string
from nltk.stem import WordNetLemmatizer


def remove_urls(text):
	#Quitar URLs
	url_pattern = re.compile(r'https?://\S+|www\.\S+')
	return url_pattern.sub(r'', text)

def remove_emoji(text):
	# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hashtags_mentions(text):
    # Expresión regular para eliminar hashtags y menciones
    text_cleaned = re.sub(r'[@#]\w+', '', text)
    # Elimina espacios adicionales que pudieran quedar tras la limpieza
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    return text_cleaned

def remove_html(text):
	#Limpiar código HTML
    return BeautifulSoup(text, "lxml").text


def remove_punctuation(text, punctuation):
    #Verificar letra a letra y juntar todo el texto
    punctuationfree = "".join([i for i in text if i not in string.punctuation]) 
    return punctuationfree

def remove_numbers(text):
    """ Remover números """
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def replace_strings(text, patterns):
	# Reemplazar ciertas cadenas de texto por otras
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text) #specifies strings or a set of strings or patterns that match it
    return text


#defining the function to remove stopwords from tokenized te
def remove_stopwords(text, stopwords):
    output= [i for i in text if i not in stopwords]
    return output

def stemming_es(token):
	# stemming the texto en español:
    es_stemmer = SnowballStemmer('spanish') # for english
    return es_stemmer.stem(token) # Para una palabra en español
     

def stemming_eng(text):
	# stemming the texto en inglés:
    eng_stemmer = PorterStemmer() # for english
    stem_text = [eng_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer_eng(text, wordnet_lemmatizer):
	#defining the function for lemmatization:
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text