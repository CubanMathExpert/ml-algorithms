import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import torch
import transformers as ppb

from transformers import BertTokenizer

import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter='\t')

dataFrame = dataFrame[['tweet', 'subtask_a']]

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
        last_hiddden_states_for_batch = model(input_ids[i:i + batch_size], attention_mask=attention_mask[i:i + batch_size])
        result.append(last_hiddden_states_for_batch)


