import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import torch
import transformers as ppb

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter='\t')

dataFrame = dataFrame[['tweet', 'subtask_a']]

dataFrame = dataFrame[:500]

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = model_class.from_pretrained(pretrained_weights)

# tokenization
tokenized = dataFrame['tweet'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# padding
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

# masking
attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

result = []
batch_size = 40

with torch.no_grad():
    for i in tqdm(range(0, input_ids.shape[0], batch_size)):
        last_hidden_states_for_batch = model(input_ids[i:i + batch_size], attention_mask=attention_mask[i:i + batch_size])
        result.append(last_hidden_states_for_batch)

last_hidden_states = torch.cat(list(map(lambda x : x.last_hidden_state, result)), dim=0)
features = last_hidden_states[:,0,:].numpy()
labels = dataFrame['subtask_a']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

#print(train_features, train_labels)

model_mnb = LogisticRegression()
model_mnb.fit(train_features, train_labels)

print(model_mnb.score(test_features,test_labels))
