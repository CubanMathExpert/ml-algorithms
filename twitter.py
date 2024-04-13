import pandas as pd
import numpy as np
import torch
import string
import transformers as ppb
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# nltk downloads
#nltk.download('punkt')
#nltk.download('stopwords')

dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter="\t") # read tsv file with pandas
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
vectorizer = TfidfVectorizer(max_features=550) 
X = vectorizer.fit_transform(dataFrame['stemmed_tweet'])
X_dense = X.toarray()
encoder = LabelEncoder()
y = encoder.fit_transform(dataFrame['subtask_a'])

# train / test splits
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, train_size=0.7, random_state=0)


# models

# naive bayes
model_mnb = MultinomialNB()
model_gnb = GaussianNB()
y_pred_mnb = model_mnb.fit(X_train, y_train).predict(X_test)
y_pred_gnb = model_gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy for Multinomial Naive Bayes: ", accuracy_score(y_test, y_pred_mnb))
print("Accuracy for Gaussian Naive Bayes: ", accuracy_score(y_test, y_pred_gnb))

# # decision tree
# model_dt = DecisionTreeClassifier()
# y_pred_dt = model_dt.fit(X_train, y_train).predict(X_test)
# print("Accuracy for Decision Tree: ", accuracy_score(y_test, y_pred_dt))

# # random forests
# model_rf = RandomForestClassifier(n_estimators=10)
# y_pred_rf = model_rf.fit(X_train, y_train).predict(X_test)
# print("Accuracy for Random Forests: ", accuracy_score(y_test, y_pred_rf))

# # SVM
# model_svm_lin = SVC(kernel='linear')
# model_svm_rbf = SVC(kernel='rbf')
# y_pred_svm_lin = model_svm_lin.fit(X_train, y_train).predict(X_test)
# y_pred_svm_rbf = model_svm_rbf.fit(X_train, y_train).predict(X_test)
# print("Accuracy for SVM linear: ", accuracy_score(y_test, y_pred_svm_lin))
# print("Accuracy for SVM rbf: ", accuracy_score(y_test, y_pred_svm_rbf))

# # neural network
# model_mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 50), max_iter=800, momentum=0.9, learning_rate_init=0.001, batch_size=300, random_state=0)
# y_pred_mlp = model_mlp.fit(X_train, y_train).predict(X_test)
# print("Accuracy for Multi Layer Perceptron: ", accuracy_score(y_test, y_pred_mlp))



##print("Classification Report:\n", classification_report(y_test, y_pred))



