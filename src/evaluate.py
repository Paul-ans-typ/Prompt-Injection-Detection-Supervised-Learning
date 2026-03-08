import os
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# We need the dataset class again for the BERT dataloader
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Call the tokenizer directly instead of using encode_plus
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    test_path = 'data/processed/test.csv'
    print("Loading test data...")
    test_df = pd.read_csv(test_path).dropna(subset=['Prompt', 'isMalicious'])
    
    X_test_texts = test_df['Prompt'].astype(str).tolist()
    y_test = test_df['isMalicious'].tolist()

    # ==========================================
    # 1. Evaluate Baseline (TF-IDF + LR)
    # ==========================================
    print("\n--- Evaluating Baseline Model (Logistic Regression) ---")
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    lr_model = joblib.load('models/baseline_lr_model.joblib')

    X_test_tfidf = vectorizer.transform(X_test_texts)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    print("Baseline Classification Report:")
    print(classification_report(y_test, y_pred_lr, digits=4))
    
    os.makedirs('notebooks', exist_ok=True)
    plot_confusion_matrix(y_test, y_pred_lr, 'Logistic Regression Confusion Matrix', 'notebooks/cm_baseline.png')
    print("Saved Baseline Confusion Matrix to 'notebooks/cm_baseline.png'")

    # ==========================================
    # 2. Evaluate Deep Learning Model (BERT)
    # ==========================================
    print("\n--- Evaluating BERT Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = 'models/bert_model/'
    
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    bert_model = BertForSequenceClassification.from_pretrained(model_dir)
    bert_model = bert_model.to(device)
    bert_model.eval()

    MAX_LEN = 256
    BATCH_SIZE = 32
    
    test_dataset = PromptDataset(X_test_texts, y_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred_bert = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).flatten()
            y_pred_bert.extend(preds.cpu().numpy())

    print("BERT Classification Report:")
    print(classification_report(y_test, y_pred_bert, digits=4))
    
    plot_confusion_matrix(y_test, y_pred_bert, 'BERT Confusion Matrix', 'notebooks/cm_bert.png')
    print("Saved BERT Confusion Matrix to 'notebooks/cm_bert.png'")

if __name__ == "__main__":
    main()