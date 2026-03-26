import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/val.csv'
    model_dir = 'models/bert_model/'
    
    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed data...")
    train_df = pd.read_csv(train_path).dropna(subset=['Prompt', 'isMalicious'])
    val_df = pd.read_csv(val_path).dropna(subset=['Prompt', 'isMalicious'])

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Hyperparameters
    MAX_LEN = 256
    BATCH_SIZE = 32  # Safely accommodated by a 16GB VRAM GPU footprint
    EPOCHS = 4
    LEARNING_RATE = 2e-5

    print("Preparing datasets and dataloaders...")
    train_dataset = PromptDataset(
        train_df['Prompt'].to_numpy(), 
        train_df['isMalicious'].to_numpy(), 
        tokenizer, 
        MAX_LEN
    )
    val_dataset = PromptDataset(
        val_df['Prompt'].to_numpy(), 
        val_df['isMalicious'].to_numpy(), 
        tokenizer, 
        MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing BERT model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Training phase
        model.train()
        total_train_loss = 0

        train_progress = tqdm(train_loader, desc=f"Training", leave=True)
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update progress bar with current loss
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []

        val_progress = tqdm(val_loader, desc=f"Validation", leave=True)
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        print("Validation Report:")
        print(classification_report(val_labels, val_preds, digits=4))

    print(f"Saving fine-tuned model and tokenizer to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Done!")

if __name__ == "__main__":
    main()