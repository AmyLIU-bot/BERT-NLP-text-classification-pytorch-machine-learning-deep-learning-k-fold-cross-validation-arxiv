import pandas as pd #use pandas to read csv file, read database
import numpy as np #using for mathmatic calculation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # for the performance metrics calculation.
import torch  #Pytoch used for model building and training. 
from torch.utils.data import Dataset, DataLoader #used to load data for Pytorch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW #download tokenizer for BERT model
from transformers import get_scheduler #for larning rate schedulers.

# Access to the data set, set file_path with the name, use pandas function to read the csv file.
file_path = 'research_abstracts.csv' #prepare for read data from csv
data = pd.read_csv(file_path) #use pandas' pd.read_csv to read raw data from csv.

# use the region file in the csv file as numerical label for BERT mdol learning and validation.
labels_map = {region: idx for idx, region in enumerate(data['Region'].unique())}  #using use each unique data in region as label. no repoeat label
data['label'] = data['Region'].map(labels_map)  #use data in region as BERT laerning mpdel's label

# Use BERT tokenizer to generate the tensor for each words and sentence. Input is the original text, output is machine-readable data. Detail explaination are in the report.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_data(texts, labels): #define tokenize_data functionl, including parameter texts and labels.
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors='pt') #using truncation, padding, and max_length to ensure the length of text matrix. 
    return encodings, torch.tensor(labels.values) #reture machine rdeable tensor.

# Create a class that used to access the data from dataset
class AbstractsDataset(Dataset):
    def __init__(self, encodings, labels): #initiate the dataset with it's encoding and labels, encoding was return in tokenizer function.
        self.encodings = encodings #save encodings as encoding attribute.
        self.labels = labels #save labels as labels attribute

    def __len__(self):
        return len(self.labels) #define len function, return the size of the dataset

    def __getitem__(self, idx): #define getitem function, get each item(dataset) and its label.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# K-Fold Cross-Validation, here setting the K; changing the K to compare it influence for the model; the technology is explained the methodology part inside the report. 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #used to change the K-folder value, increase mdel accuracy by decreasing overfitting
abstracts = data['Abstract']#make sure the traning dataset abstracts read from 'Abstract' column. 
labels = data['label']#assign label with the value in region in csv which was assigned in data['label'] before.

#Batch size and epoch size setting; these is explained in the report
num_epochs = 10 #setting epoch size, how many times each set of data will be trained in one fold.
batch_size = 16 #setting batch_size, the size of data that will be passed to the model togther. 
fold = 1 #start from fold 1.

#initialize the storage for store the output including performance metrics, and confusion matrics (the value that used to calculate them)
all_predictions = [] # used to store all prediction (predict the classfication of each abstracts)
all_true_labels = [] # used to store the real label (that assigned in CSV)
overall_conf_matrix = np.zeros((len(labels_map), len(labels_map))) # used to calculate the confusion matrix, detail explianation are in report.

#train and validate the dataset for each fold with BERT model.
for train_idx, val_idx in kf.split(abstracts, labels): #split abstracts into different fold, and different trained and validate dataset.
    print(f"Starting Fold {fold}") # print a signal to show coder, fold one start.
    train_texts, val_texts = abstracts.iloc[train_idx], abstracts.iloc[val_idx] #geting train and validate subset
    train_labels, val_labels = labels.iloc[train_idx], labels.iloc[val_idx] #getting train and validate labels.

    train_encodings, train_labels_tensor = tokenize_data(train_texts, train_labels) #use tokenize_data function created before to encode subset
    val_encodings, val_labels_tensor = tokenize_data(val_texts, val_labels)#use tokenize_data function created before to encode label

    train_dataset = AbstractsDataset(train_encodings, train_labels_tensor) #create training subset
    val_dataset = AbstractsDataset(val_encodings, val_labels_tensor) #create validate subset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #create train subset loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)#create validate subset loader

    # model initialization
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_map)) #use BERT function to nitialize the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #check GPU availibility
    model.to(device) #move model to the target device
    optimizer = AdamW(model.parameters(), lr=5e-5) #initialize AdamW optimizer
    num_training_steps = len(train_loader) * num_epochs #calculate overall steps for training
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps) #create learning rate scheduler.

    for epoch in range(num_epochs): #make sure each epoch is running
        #start training
        model.train() #train the BERT model, training mode
        for batch in train_loader: #pass daba by batch size
            batch = {k: v.to(device) for k, v in batch.items()} #pass dava into device
            outputs = model(**batch) #get model output
            loss = outputs.loss #calculate model los
            optimizer.zero_grad() #clear gradients
            loss.backward() #pass backword
            optimizer.step() #renew, update weights
            lr_scheduler.step() #update learning rate

        # validate model
        model.eval() #validate model, evaluation mode
        predictions, true_labels = [], [] #initialize prediction and label
        with torch.no_grad(): #ignore gradient computation
            for batch in val_loader: #each batch in the loaded data
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch) #predict output by BERT model
                preds = torch.argmax(outputs.logits, dim=1) #get prediction label
                predictions.extend(preds.cpu().numpy()) #save prediction label
                true_labels.extend(batch['labels'].cpu().numpy()) #save ture label

        # Epoch performance metrics
        acc = accuracy_score(true_labels, predictions) #calculate accuracy score for each epoch
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted') #calculate precision, recall, f1 score for each epoch
        print(f"Fold {fold}; Epoch {epoch + 1}: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}") #print out performence metrics for each epoch

    # Fold performance metrics
    fold_acc = accuracy_score(true_labels, predictions) #calculate accuracy score for each fold
    fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')) #calculate precision, recall, f1 score for each fold
    fold_conf_matrix = confusion_matrix(true_labels, predictions) #calculate confusion matrix for each fold
    print(f"Fold {fold} Final Metrics: Accuracy={fold_acc:.4f}, Precision={fold_precision:.4f}, Recall={fold_recall:.4f}, F1-Score={fold_f1:.4f}")#print out performence metrics for each fold
    print(f"Fold {fold} Confusion Matrix:\n{fold_conf_matrix}")#print out confusion matrix fo each fold 

    # save current output and true value for current fold
    all_predictions.extend(predictions) #add current prediction output into overall prediction list
    all_true_labels.extend(true_labels) #add current true label into overall prediction list
    overall_conf_matrix += fold_conf_matrix #add current confusion matrix into oberall confusion matrix.

    # save model weight in .pth; save model
    save_path = f'bert_model_fold_{fold}.pth' #save with .pth
    torch.save(model.state_dict(), save_path) #save current dictionary into target file

    fold += 1

# output overall cperformamce Metrics amd cpnfusion matrix.
overall_acc = accuracy_score(all_true_labels, all_predictions)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
print("\nOverall Final Metrics:")
print(f"Accuracy: {overall_acc:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-Score: {overall_f1:.4f}")
print(f"Overall Confusion Matrix:\n{overall_conf_matrix}")