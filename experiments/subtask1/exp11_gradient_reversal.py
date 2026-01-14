"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 11 - Gradient Reversal
================================================================================
Description: Adversarial debiasing using Gradient Reversal Layer (GRL).
             
Idea: Train a secondary classifier to predict plausibility from hidden states.
      Use gradient reversal to make main model representations INVARIANT
      to plausibility, forcing it to focus only on logical structure.
             
Paper: "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
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
from torch.autograd import Function
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
    OUTPUT_DIR = '/kaggle/working/deberta_exp11'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    
    # Gradient Reversal parameters
    GRL_LAMBDA = 0.5  # Strength of adversarial loss
    LABEL_SMOOTHING = 0.1
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(Function):
    """Gradient Reversal Layer - reverses gradient during backpropagation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ============================================================================
# 5. ADVERSARIAL DEBIASING MODEL
# ============================================================================
class AdversarialDebiasingModel(nn.Module):
    """
    DeBERTa with adversarial debiasing:
    - Main classifier: predicts validity (what we want)
    - Adversarial classifier: predicts plausibility (what we want to remove)
    
    Using gradient reversal, we train the encoder to fool the adversarial
    classifier, making representations invariant to plausibility.
    """
    
    def __init__(self, model_name, grl_lambda=0.5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        
        # Main task: validity classification
        self.validity_classifier = nn.Linear(hidden_size, 2)
        
        # Adversarial task: plausibility classification (with GRL)
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.plausibility_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        
        # Main task
        validity_logits = self.validity_classifier(pooled)
        
        # Adversarial task (with gradient reversal)
        reversed_pooled = self.grl(pooled)
        plausibility_logits = self.plausibility_classifier(reversed_pooled)
        
        return validity_logits, plausibility_logits

# ============================================================================
# 6. DATASET
# ============================================================================
class AdversarialDataset(Dataset):
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
            result['validity'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
            result['plausibility'] = torch.tensor(1 if item['plausibility'] else 0, dtype=torch.long)
        
        return result

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        validity = batch['validity'].to(device)
        plausibility = batch['plausibility'].to(device)
        
        optimizer.zero_grad()
        
        validity_logits, plausibility_logits = model(input_ids, attention_mask)
        
        # Main task loss
        validity_loss = ce_fn(validity_logits, validity)
        
        # Adversarial loss (gradient reversed automatically by GRL)
        adversarial_loss = ce_fn(plausibility_logits, plausibility)
        
        # Combined loss
        loss = validity_loss + Config.GRL_LAMBDA * adversarial_loss
        
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
            validity = batch['validity'].to(device)
            
            validity_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(validity_logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(validity.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 11: Gradient Reversal Adversarial Debiasing")
    print(f"  GRL Lambda: {Config.GRL_LAMBDA}")
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
        AdversarialDataset(train_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        AdversarialDataset(val_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE * 2
    )
    
    # Model
    print(f"\n🚀 Training with Gradient Reversal...")
    model = AdversarialDebiasingModel(Config.MODEL_NAME, grl_lambda=Config.GRL_LAMBDA).to(device)
    
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
            
            validity_logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(validity_logits, dim=-1).item()
            predictions.append({'id': item['id'], 'validity': bool(pred)})
    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
