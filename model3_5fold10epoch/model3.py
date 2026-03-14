#batch: 16; fold: 5; epoch: 10
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler

# Load dataset
file_path = 'research_abstracts.csv'
data = pd.read_csv(file_path)

# Map regions to numeric labels
labels_map = {region: idx for idx, region in enumerate(data['Region'].unique())}
data['label'] = data['Region'].map(labels_map)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
    return encodings, torch.tensor(labels.values)

# Dataset Class
class AbstractsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Model Setup
num_labels = len(labels_map)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Changed n_splits to 5
abstracts = data['Abstract']
labels = data['label']

# Variables for overall metrics
all_predictions = []
all_true_labels = []
all_conf_matrix = np.zeros((num_labels, num_labels))

# Training Loop with K-Fold
num_epochs = 10
fold = 1
for train_idx, val_idx in kf.split(abstracts, labels):
    print(f"Starting Fold {fold}")
    train_texts, val_texts = abstracts.iloc[train_idx], abstracts.iloc[val_idx]
    train_labels, val_labels = labels.iloc[train_idx], labels.iloc[val_idx]

    train_encodings, train_labels_tensor = tokenize_data(train_texts, train_labels)
    val_encodings, val_labels_tensor = tokenize_data(val_texts, val_labels)

    train_dataset = AbstractsDataset(train_encodings, train_labels_tensor)
    val_dataset = AbstractsDataset(val_encodings, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Scheduler for each fold
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train Model
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print(f"Fold {fold}, Epoch {epoch + 1} completed.")

    # Evaluate Model
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    # Calculate metrics for this fold
    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)

    print(f"Fold {fold} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accumulate metrics for the overall confusion matrix
    all_predictions.extend(predictions)
    all_true_labels.extend(true_labels)
    all_conf_matrix += conf_matrix

    fold += 1

# Overall Results
overall_acc = accuracy_score(all_true_labels, all_predictions)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
overall_conf_matrix = confusion_matrix(all_true_labels, all_predictions)

print("\nOverall Results:")
print(f"Overall Accuracy: {overall_acc:.4f}")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1-Score: {overall_f1:.4f}")

print("\nOverall Confusion Matrix:")
print(overall_conf_matrix)

# Save Model
model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')