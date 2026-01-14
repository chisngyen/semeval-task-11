"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 9 - Exp5 + Exp6 Combined
================================================================================
Description: Combine the best from Exp5 and Exp6:
             - From Exp5: Cosine LR, Label Smoothing, Longer Warmup
             - From Exp6: R-Drop Regularization
             
Difficulty: Medium
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION - COMBINED BEST OF EXP5 + EXP6
# ============================================================================
class Config:
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_exp9'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    # From Exp5: Better Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6  # Lower LR from exp5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10  # More epochs from exp5
    WARMUP_RATIO = 0.15  # Longer warmup from exp5
    LABEL_SMOOTHING = 0.1  # Label smoothing from exp5
    LR_SCHEDULER = 'cosine'  # Cosine schedule from exp5
    
    # From Exp6: R-Drop
    RDROP_ALPHA = 0.7  # R-Drop weight from exp6
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. DATASET
# ============================================================================
class SyllogismDataset(Dataset):
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
        
        return result

# ============================================================================
# 5. R-DROP TRAINING WITH BETTER HPARAMS
# ============================================================================
def compute_kl_loss(p, q):
    """Compute symmetric KL divergence."""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)  # From exp5
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # R-Drop: Two forward passes with different dropout (from exp6)
        logits1 = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits2 = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Cross-entropy loss (average)
        ce_loss = (ce_fn(logits1, labels) + ce_fn(logits2, labels)) / 2
        
        # KL divergence loss for consistency
        kl_loss = compute_kl_loss(logits1, logits2)
        
        # Combined loss
        loss = ce_loss + Config.RDROP_ALPHA * kl_loss
        
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
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 9: Exp5 + Exp6 Combined")
    print("  - Cosine LR, Label Smoothing, Longer Warmup (from Exp5)")
    print("  - R-Drop Regularization (from Exp6)")
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
    print(f"   Train split: {len(train_split)}, Val split: {len(val_split)}")
    
    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_loader = DataLoader(
        SyllogismDataset(train_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE * 2
    )
    
    # Model
    print(f"\n🚀 Training with Combined Strategy...")
    print(f"   LR: {Config.LEARNING_RATE}, Scheduler: {Config.LR_SCHEDULER}")
    print(f"   Label Smoothing: {Config.LABEL_SMOOTHING}")
    print(f"   R-Drop Alpha: {Config.RDROP_ALPHA}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, problem_type="single_label_classification"
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    num_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_scheduler(
        Config.LR_SCHEDULER, optimizer,
        num_warmup_steps=int(num_steps * Config.WARMUP_RATIO),
        num_training_steps=num_steps
    )
    
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
            model.save_pretrained(os.path.join(Config.OUTPUT_DIR, 'best_model'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("   Early stopping!")
                break
    
    # Load best model
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(Config.OUTPUT_DIR, 'best_model')
    ).to(device)
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
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append({'id': item['id'], 'validity': bool(pred)})
    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    # Cleanup
    import shutil
    if os.path.exists(os.path.join(Config.OUTPUT_DIR, 'best_model')):
        shutil.rmtree(os.path.join(Config.OUTPUT_DIR, 'best_model'))
    
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print(f"   Valid: {sum(1 for p in predictions if p['validity'])}")
    print(f"   Invalid: {sum(1 for p in predictions if not p['validity'])}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
