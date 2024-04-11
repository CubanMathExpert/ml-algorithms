import pandas as pd
import numpy as np
import torch as pt
import transformers as ppb
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
nltk.download('punkt')

dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter="\t")

# stem the file
stemmer = PorterStemmer()
def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

dataFrame['stemmed_tweet'] = dataFrame['tweet'].apply(stem_text) # add column 'stemmed_tweet' to dataFrame

vectorizer = TfidfVectorizer(max_features=10000)
x = vectorizer.fit_transform(dataFrame['stemmed_tweet'])

print(x)












