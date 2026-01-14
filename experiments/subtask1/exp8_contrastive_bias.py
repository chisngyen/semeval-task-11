"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 8 - Contrastive + Bias Penalty
================================================================================
Description: DeBERTa-v3-large with contrastive learning and bias penalty.
Difficulty: Hard
Hardware: Kaggle H100 GPU
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'protobuf==3.20.3'])
    packages = ['transformers>=4.36.0', 'accelerate>=0.25.0']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

install_packages()

# ============================================================================
# 2. IMPORTS
# ============================================================================
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
class Config:
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_exp8'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    
    CONTRASTIVE_WEIGHT = 0.1
    BIAS_PENALTY_WEIGHT = 0.2
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. CONTRASTIVE MODEL
# ============================================================================
class ContrastiveBiasModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(self.dropout(pooled))
        embeddings = F.normalize(self.projection(pooled), dim=-1)
        
        return logits, embeddings

# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================
def contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss."""
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create masks without inplace operations
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels_eq.float() * diag_mask.float()
    
    exp_sim = torch.exp(sim_matrix) * diag_mask.float()
    
    positives = (exp_sim * positive_mask).sum(dim=1)
    all_pairs = exp_sim.sum(dim=1)
    
    loss = -torch.log((positives + 1e-8) / (all_pairs + 1e-8))
    valid_mask = positive_mask.sum(dim=1) > 0
    
    if valid_mask.sum() > 0:
        return loss[valid_mask].mean()
    return torch.tensor(0.0, device=embeddings.device)

def bias_penalty_loss(logits, labels, plausibility):
    """Penalty for different confidence across plausibility conditions."""
    probs = F.softmax(logits, dim=-1)
    penalty = torch.tensor(0.0, device=logits.device)
    
    for validity in [0, 1]:
        val_mask = (labels == validity)
        if val_mask.sum() < 2:
            continue
        
        plaus_true = val_mask & (plausibility == 1)
        plaus_false = val_mask & (plausibility == 0)
        
        if plaus_true.sum() > 0 and plaus_false.sum() > 0:
            conf_true = probs[plaus_true, validity].mean()
            conf_false = probs[plaus_false, validity].mean()
            penalty = penalty + torch.abs(conf_true - conf_false)
    
    return penalty

# ============================================================================
# 6. DATASET
# ============================================================================
class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Determine if the following syllogism is logically valid:\n\n{item['syllogism']}"
        
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if not self.is_test:
            result['labels'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
            result['plausibility'] = torch.tensor(1 if item['plausibility'] else 0, dtype=torch.long)
        
        return result

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        plausibility = batch['plausibility'].to(device)
        
        optimizer.zero_grad()
        
        logits, embeddings = model(input_ids, attention_mask)
        
        # Combined loss
        ce_loss = ce_fn(logits, labels)
        cont_loss = contrastive_loss(embeddings, labels)
        bias_loss = bias_penalty_loss(logits, labels, plausibility)
        
        loss = ce_loss + Config.CONTRASTIVE_WEIGHT * cont_loss + Config.BIAS_PENALTY_WEIGHT * bias_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 8: Contrastive + Bias Penalty (Custom Loop)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(
        train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED
    )
    
    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_loader = DataLoader(
        ContrastiveDataset(train_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        ContrastiveDataset(val_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE * 2
    )
    
    # Model
    print(f"\n🚀 Training with Contrastive (w={Config.CONTRASTIVE_WEIGHT}) + Bias Penalty (w={Config.BIAS_PENALTY_WEIGHT})...")
    model = ContrastiveBiasModel(Config.MODEL_NAME).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    num_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=int(num_steps * Config.WARMUP_RATIO), num_training_steps=num_steps)
    
    best_acc = 0
    patience = 3
    no_improve = 0
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"   Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, 'best_model.pt'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("   Early stopping!")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR, 'best_model.pt')))
    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    
    # Inference
    print("\n🔮 Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting"):
            text = f"Determine if the following syllogism is logically valid:\n\n{item['syllogism']}"
            encoding = tokenizer(text, max_length=Config.MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append({'id': item['id'], 'validity': bool(pred)})
    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print(f"   Valid: {sum(1 for p in predictions if p['validity'])}")
    print(f"   Invalid: {sum(1 for p in predictions if not p['validity'])}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
