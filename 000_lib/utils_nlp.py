
from collections import Counter
import nltk

from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

"""
# reverse mapping
# if you do it smarter you can store it as a list
idx2word = {v:k for k, v in word2idx.items()}
"""

def wordToIdx(df):
    # populate word2idx
    # convert documents into sequences of ints / ids / indices
    idx = 0
    word2idx = {}
    tokenized_docs = []
    for doc in df['text']:
        words = word_tokenize(doc.lower())
        doc_as_int = []
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1

            # save for later
            doc_as_int.append(word2idx[word])
        tokenized_docs.append(doc_as_int)

    return word2idx, doc_as_int, tokenized_docs

# Lets check keywords
def get_most_common_keywords(df, label="title", amount=20):

    return Counter(" ".join(df[label]).split()).most_common(amount)

def label_sentiment(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    if neg > pos:
        return -neg
    elif pos > neg:
        return pos
    return 0