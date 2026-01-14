"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 15 - Focus on Reducing TCE
================================================================================
Description: Focus on REDUCING CONTENT EFFECT while maintaining accuracy:
             1. Balanced sampling: equal (validity, plausibility) groups per batch
             2. Adversarial debiasing with higher lambda
             3. Contrastive loss between same-validity different-plausibility
             4. Logic-aware prompt (from Exp12)
             
Goal: TCE < 3.0 (currently best is 4.28)
             
Difficulty: Hard
Hardware: Kaggle H100 GPU
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys
import re
import random

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
from torch.utils.data import Dataset, DataLoader, Sampler
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
    OUTPUT_DIR = '/kaggle/working/deberta_exp15'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    
    # Strong debiasing
    GRL_LAMBDA = 1.0  # Higher for stronger debiasing (was 0.3)
    CONTRASTIVE_WEIGHT = 0.3  # Pull same-validity pairs closer
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. BALANCED BATCH SAMPLER
# ============================================================================
class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has balanced (validity, plausibility) combinations.
    This prevents the model from learning plausibility as a shortcut.
    """
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        
        # Group indices by (validity, plausibility)
        self.groups = {
            (True, True): [],
            (True, False): [],
            (False, True): [],
            (False, False): []
        }
        
        for idx, item in enumerate(data):
            key = (item['validity'], item['plausibility'])
            self.groups[key].append(idx)
        
        print(f"   Balanced sampler groups:")
        for key, indices in self.groups.items():
            print(f"      {key}: {len(indices)} samples")
    
    def __iter__(self):
        # Shuffle each group
        for key in self.groups:
            random.shuffle(self.groups[key])
        
        # Create balanced batches
        batches = []
        samples_per_group = self.batch_size // 4
        
        # Calculate how many full batches we can make
        min_group_size = min(len(g) for g in self.groups.values())
        num_batches = min_group_size // samples_per_group
        
        for batch_idx in range(num_batches):
            batch = []
            for key in self.groups:
                start = batch_idx * samples_per_group
                end = start + samples_per_group
                batch.extend(self.groups[key][start:end])
            random.shuffle(batch)
            batches.append(batch)
        
        # Shuffle batches
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        min_group_size = min(len(g) for g in self.groups.values())
        samples_per_group = self.batch_size // 4
        return min_group_size // samples_per_group

# ============================================================================
# 5. GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ============================================================================
# 6. DEBIASED MODEL
# ============================================================================
class DebiasedModel(nn.Module):
    def __init__(self, model_name, grl_lambda=1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        
        # Main task: validity
        self.validity_head = nn.Linear(hidden_size, 2)
        
        # Adversarial: plausibility (with strong GRL)
        self.grl = GRL(lambda_=grl_lambda)
        self.plausibility_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        
        validity_logits = self.validity_head(pooled)
        
        reversed_pooled = self.grl(pooled)
        plausibility_logits = self.plausibility_head(reversed_pooled)
        
        return validity_logits, plausibility_logits, pooled

# ============================================================================
# 7. DATASET
# ============================================================================
class DebiasDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Analyze logical validity only, ignore real-world truth:\n\n{item['syllogism']}"
        
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
# 8. CONTRASTIVE LOSS FOR DEBIASING
# ============================================================================
def compute_contrastive_debiasing_loss(embeddings, validity, plausibility):
    """
    Pull together samples with same validity but different plausibility.
    This teaches the model that plausibility doesn't matter for validity.
    """
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Positive pairs: same validity, different plausibility
    validity_eq = validity.unsqueeze(0) == validity.unsqueeze(1)
    plausibility_neq = plausibility.unsqueeze(0) != plausibility.unsqueeze(1)
    positive_mask = (validity_eq & plausibility_neq).float()
    
    # Diagonal mask
    diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask * diag_mask.float()
    
    # Loss: maximize similarity of positive pairs (same val, diff plaus)
    positive_sims = (sim_matrix * positive_mask).sum()
    num_positives = positive_mask.sum()
    
    if num_positives > 0:
        return -positive_sims / num_positives  # Maximize similarity = minimize negative
    return torch.tensor(0.0, device=embeddings.device)

# ============================================================================
# 9. TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        validity = batch['validity'].to(device)
        plausibility = batch['plausibility'].to(device)
        
        optimizer.zero_grad()
        
        val_logits, plaus_logits, embeddings = model(input_ids, attention_mask)
        
        # Validity loss (main task)
        val_loss = ce_fn(val_logits, validity)
        
        # Adversarial plausibility loss (GRL reverses gradients)
        adv_loss = ce_fn(plaus_logits, plausibility)
        
        # Contrastive debiasing loss
        contrast_loss = compute_contrastive_debiasing_loss(embeddings, validity, plausibility)
        
        loss = val_loss + Config.GRL_LAMBDA * adv_loss + Config.CONTRASTIVE_WEIGHT * contrast_loss
        
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
            
            val_logits, _, _ = model(input_ids, attention_mask)
            preds = torch.argmax(val_logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(validity.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 10. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 15: Focus on Reducing TCE")
    print("  - Balanced batch sampling")
    print("  - Strong gradient reversal (lambda={})".format(Config.GRL_LAMBDA))
    print("  - Contrastive debiasing")
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
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = DebiasDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = DebiasDataset(val_split, tokenizer, Config.MAX_LENGTH)
    
    # Balanced sampler for training
    train_sampler = BalancedBatchSampler(train_split, Config.BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2)
    
    # Model
    print(f"\n🚀 Training Debiased Model...")
    model = DebiasedModel(Config.MODEL_NAME, grl_lambda=Config.GRL_LAMBDA).to(device)
    
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
    
    # Inference
    print("\n🔮 Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting"):
            text = f"Analyze logical validity only, ignore real-world truth:\n\n{item['syllogism']}"
            encoding = tokenizer(text, max_length=Config.MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            val_logits, _, _ = model(input_ids, attention_mask)
            pred = torch.argmax(val_logits, dim=-1).item()
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
</Parameter>
<parameter name="Complexity">8
