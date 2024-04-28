import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder  # Added this import
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import transformers as ppb
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Create device for hardware acceleration

# Load data
dataFrame = pd.read_csv("source/offenseval-training-v1.tsv", delimiter='\t')
dataFrame = dataFrame[['tweet', 'subtask_a']]
dataFrame = dataFrame[:1000]

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
dataFrame['label'] = label_encoder.fit_transform(dataFrame['subtask_a'])

# Load pre-trained BERT tokenizer and model
tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')
model = ppb.BertModel.from_pretrained('bert-base-uncased').to(device)

# Tokenization
tokenized = dataFrame['tweet'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Padding
max_len = max(len(token) for token in tokenized)
padded = np.array([token + [0]*(max_len-len(token)) for token in tokenized])

# Attention masks
attention_mask = np.where(padded != 0, 1, 0)

# Convert to tensors
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(dataFrame['label'])

# Split data into training and testing sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(input_ids, attention_mask, labels, train_size=0.7)

# Send data to GPU for acceleration
train_inputs, train_masks, train_labels = train_inputs.to(device), train_masks.to(device), train_labels.to(device)
test_inputs, test_masks, test_labels = test_inputs.to(device), test_masks.to(device), test_labels.to(device)

# Define batch size
batch_size = 8

# Create data loaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Fine-tuning BERT using logistic regression
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

epochs = 3
for epoch in range(epochs):
    for batch in tqdm(train_loader):
        batch_inputs, batch_masks, batch_labels = batch
        optimizer.zero_grad()
        outputs = model(batch_inputs, attention_mask=batch_masks)
        logits = outputs.last_hidden_state[:, 0, :]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(test_inputs, attention_mask=test_masks)
    logits = outputs.last_hidden_state[:, 0, :]
    preds = torch.argmax(logits, axis=1)
    accuracy = np.mean(preds.cpu().numpy() == test_labels.cpu().numpy())  # Move tensors to CPU for numpy conversion
    print("Accuracy:", accuracy)