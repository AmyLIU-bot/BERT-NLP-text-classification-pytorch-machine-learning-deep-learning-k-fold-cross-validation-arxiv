import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler

# Access to the data set, set file_path with the name, use pandas function to read the csv file.
file_path = 'research_abstracts.csv'
data = pd.read_csv(file_path)

# Create numerical label with the field label inside the CSV file.
labels_map = {region: idx for idx, region in enumerate(data['Region'].unique())}
data['label'] = data['Region'].map(labels_map)

# Use BERT tokenizer to generate the tensor for each words and sentence. Input is the original text, output is machine-readable data. Detail explaination are in the report.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
    return encodings, torch.tensor(labels.values)

# Create a class for the abstract dataset
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

# K-Fold Cross-Validation, here setting the K; changing the K to compare it influence for the model; the technology is explained the methodology part inside the report. 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
abstracts = data['Abstract']
labels = data['label']

#Batch size and epoch size setting; these is explained in the report
num_epochs = 10
batch_size = 16
fold = 1

# Overall Metrics Storage
all_predictions = []
all_true_labels = []
overall_conf_matrix = np.zeros((len(labels_map), len(labels_map)))

for train_idx, val_idx in kf.split(abstracts, labels):
    print(f"Starting Fold {fold}")
    train_texts, val_texts = abstracts.iloc[train_idx], abstracts.iloc[val_idx]
    train_labels, val_labels = labels.iloc[train_idx], labels.iloc[val_idx]

    train_encodings, train_labels_tensor = tokenize_data(train_texts, train_labels)
    val_encodings, val_labels_tensor = tokenize_data(val_texts, val_labels)

    train_dataset = AbstractsDataset(train_encodings, train_labels_tensor)
    val_dataset = AbstractsDataset(val_encodings, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model and Optimizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_map))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Validation Phase
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        # Metrics for this Epoch
        acc = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        print(f"Fold {fold}; Epoch {epoch + 1}: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    # Metrics for the Fold
    fold_acc = accuracy_score(true_labels, predictions)
    fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    fold_conf_matrix = confusion_matrix(true_labels, predictions)
    print(f"Fold {fold} Final Metrics: Accuracy={fold_acc:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}, F1-Score={fold_f1:.4f}")
    print(f"Fold {fold} Confusion Matrix:\n{fold_conf_matrix}")

    # Save Fold Metrics
    all_predictions.extend(predictions)
    all_true_labels.extend(true_labels)
    overall_conf_matrix += fold_conf_matrix

    # Save Model Weights in .pth
    save_path = f'bert_model_fold_{fold}.pth'
    torch.save(model.state_dict(), save_path)

    fold += 1

# Overall Metrics
overall_acc = accuracy_score(all_true_labels, all_predictions)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
print("\nOverall Final Metrics:")
print(f"Accuracy: {overall_acc:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-Score: {overall_f1:.4f}")
print(f"Overall Confusion Matrix:\n{overall_conf_matrix}")