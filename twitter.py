import pandas as pd
import numpy as np
import torch
import transformers as ppb
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
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
X = vectorizer.fit_transform(dataFrame['stemmed_tweet'])
X_dense = X.toarray()
encoder = LabelEncoder()
y = encoder.fit_transform(dataFrame['subtask_a'])

# train / test splits
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.7, random_state=0)

model_nb = GaussianNB()

y_pred = model_nb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


#train_test_split fait le le 25-75 split test et train data??
# ensuite je passe au model .predict(les donnes ?)
#x is the input and y the target values.














