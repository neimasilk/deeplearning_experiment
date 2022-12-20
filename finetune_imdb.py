from transformers import XLMRobertaForSequenceClassification, AdamW
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2, device_map="auto", load_in_8bit=True)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# buat dataset loader

#baca dan split 
from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

# pisah antara train dan validasi
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

from transformers import XLMRobertaTokenizerFast
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)


from tensorboardX import SummaryWriter

# create a SummaryWriter
writer = SummaryWriter(logdir)


from torch.utils.data import DataLoader
from transformers import XLMRobertaForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', device_map="auto")
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
  print(f'epoch {epoch}')
  for batch in train_loader:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    loss.backward()
    optim.step()

  # set the model to evaluation mode
  model.eval()

  # initialize variables to store true labels and predictions
  true_labels = []
  predictions = []

  # iterate through the validation data
  for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # make predictions
    with torch.no_grad():
      outputs = model(input_ids, attention_mask=attention_mask)
      logits = outputs[0]
      batch_predictions = logits.argmax(dim=1).flatten().tolist()

    # store true labels and predictions
    true_labels.extend(labels.flatten().tolist())
    predictions.extend(batch_predictions)

  # compute evaluation metrics
  accuracy = accuracy_score(true_labels, predictions)
  precision = precision_score(true_labels, predictions)
  recall = recall_score(true_labels, predictions)
  f1 = f1_score(true_labels, predictions)

  # log the evaluation metrics to TensorBoard
  writer.add_scalar('accuracy', accuracy, epoch)
  writer.add_scalar('precision', precision, epoch)
  writer.add_scalar('recall', recall, epoch)
  writer.add_scalar('f1', f1, epoch)


