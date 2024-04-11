import pandas as pd
import numpy as np
import torch
import string
import transformers as ppb
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# nltk downloads
#nltk.download('punkt')
#nltk.download('stopwords')

dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter="\t")
stop_words = set(stopwords.words('english'))  # set of all the english stopwords

# stem the file
stemmer = PorterStemmer()
def stem_text(text):
    text = text.lower() # put text in lower case
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    tokens = nltk.word_tokenize(text)

    clean_tokens = [token for token in tokens if token not in  stop_words] # tokens without stop words
    stemmed_tokens = [stemmer.stem(token) for token in clean_tokens] # stem the tokens without stop words

    return ' '.join(stemmed_tokens)

dataFrame['stemmed_tweet'] = dataFrame['tweet'].apply(stem_text) # add column 'stemmed_tweet' to dataFrame

# tokenization
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(dataFrame['stemmed_tweet'])
X_dense = X.toarray()
encoder = LabelEncoder()
y = encoder.fit_transform(dataFrame['subtask_a'])

# train / test splits
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.7, random_state=0)

# models
model_nb = GaussianNB()

# training and prediction
y_pred = model_nb.fit(X_train, y_train).predict(X_test)


# check accuracy and full report 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



