"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 13 - Exp9 Improved
================================================================================
Description: Improve on Exp9 (best so far = 35.19) with:
             1. R-Drop + Label Smoothing + Cosine LR (from Exp9)
             2. Gradient Reversal for plausibility debiasing (from Exp11)
             3. Longer training with lower LR
             4. Test-time augmentation (TTA)
             
Goal: Reduce TCE while maintaining high accuracy
             
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
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION - IMPROVED FROM EXP9
# ============================================================================
class Config:
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_exp13'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    # From Exp9 (best)
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-6  # Even lower LR
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 12  # More epochs
    WARMUP_RATIO = 0.2  # Longer warmup
    LABEL_SMOOTHING = 0.1
    RDROP_ALPHA = 0.7
    
    # New: Gradient Reversal for debiasing (from Exp11)
    GRL_LAMBDA = 0.3  # Lower to not hurt accuracy too much
    
    # Test-time augmentation
    TTA_ENABLED = True
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(Function):
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
# 5. IMPROVED MODEL WITH DEBIASING
# ============================================================================
class ImprovedDeBERTa(nn.Module):
    """
    DeBERTa with:
    - Main classifier for validity
    - Adversarial classifier for plausibility (with GRL)
    """
    def __init__(self, model_name, grl_lambda=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        
        # Main task: validity
        self.validity_classifier = nn.Linear(hidden_size, 2)
        
        # Adversarial: plausibility (with GRL)
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.plausibility_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        
        validity_logits = self.validity_classifier(pooled)
        
        reversed_pooled = self.grl(pooled)
        plausibility_logits = self.plausibility_classifier(reversed_pooled)
        
        return validity_logits, plausibility_logits

# ============================================================================
# 6. DATASET
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
            result['validity'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
            result['plausibility'] = torch.tensor(1 if item['plausibility'] else 0, dtype=torch.long)
        
        return result

# ============================================================================
# 7. R-DROP + DEBIASING TRAINING
# ============================================================================
def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    return (p_loss + q_loss) / 2

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
        
        # R-Drop: Two forward passes
        val_logits1, plaus_logits1 = model(input_ids, attention_mask)
        val_logits2, plaus_logits2 = model(input_ids, attention_mask)
        
        # Validity loss (main task)
        ce_loss = (ce_fn(val_logits1, validity) + ce_fn(val_logits2, validity)) / 2
        
        # R-Drop KL consistency
        kl_loss = compute_kl_loss(val_logits1, val_logits2)
        
        # Adversarial loss (debiasing)
        adv_loss = (ce_fn(plaus_logits1, plausibility) + ce_fn(plaus_logits2, plausibility)) / 2
        
        # Combined loss
        loss = ce_loss + Config.RDROP_ALPHA * kl_loss + Config.GRL_LAMBDA * adv_loss
        
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
            
            val_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(val_logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(validity.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 8. TEST-TIME AUGMENTATION
# ============================================================================
def predict_with_tta(model, tokenizer, item, device):
    """
    Test-time augmentation: average predictions with different prompts.
    """
    prompts = [
        f"Determine if the following syllogism is logically valid:\n\n{item['syllogism']}",
        f"Is this syllogism valid or invalid? Focus on logical structure, not content.\n\n{item['syllogism']}",
        f"Analyze the logical validity:\n\n{item['syllogism']}\n\nValid or Invalid?",
    ]
    
    all_probs = []
    
    for prompt in prompts:
        encoding = tokenizer(prompt, max_length=Config.MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        val_logits, _ = model(input_ids, attention_mask)
        probs = F.softmax(val_logits, dim=-1)
        all_probs.append(probs)
    
    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0)
    pred = torch.argmax(avg_probs, dim=-1).item()
    
    return pred

# ============================================================================
# 9. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 13: Exp9 Improved (R-Drop + Debiasing + TTA)")
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
        SyllogismDataset(train_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH),
        batch_size=Config.BATCH_SIZE * 2
    )
    
    # Model
    print(f"\n🚀 Training Improved Model...")
    print(f"   R-Drop: {Config.RDROP_ALPHA}, GRL: {Config.GRL_LAMBDA}")
    print(f"   LR: {Config.LEARNING_RATE}, Epochs: {Config.NUM_EPOCHS}")
    
    model = ImprovedDeBERTa(Config.MODEL_NAME, grl_lambda=Config.GRL_LAMBDA).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    num_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=int(num_steps * Config.WARMUP_RATIO), num_training_steps=num_steps)
    
    best_acc = 0
    patience = 4
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
    
    # Inference with optional TTA
    print(f"\n🔮 Generating predictions (TTA={Config.TTA_ENABLED})...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting"):
            if Config.TTA_ENABLED:
                pred = predict_with_tta(model, tokenizer, item, device)
            else:
                text = f"Determine if the following syllogism is logically valid:\n\n{item['syllogism']}"
                encoding = tokenizer(text, max_length=Config.MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                val_logits, _ = model(input_ids, attention_mask)
                pred = torch.argmax(val_logits, dim=-1).item()
            
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
