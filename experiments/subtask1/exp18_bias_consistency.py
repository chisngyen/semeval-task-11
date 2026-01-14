"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 18 - Bias-Augmented Consistency (BCT)
================================================================================
Description: Based on "Bias-Augmented Consistency Training" (2024)
             Train model to generate CONSISTENT predictions regardless of
             whether prompt contains biasing content or not.
             
Method: For each example, create two versions:
        1. Original with content
        2. Neutralized without plausibility cues
        Force model to predict same label for both
             
Goal: Make predictions INVARIANT to content plausibility
             
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed, get_scheduler
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
    OUTPUT_DIR = '/kaggle/working/deberta_exp18'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 384
    
    BATCH_SIZE = 8  # Smaller because we process pairs
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    LABEL_SMOOTHING = 0.1
    
    # BCT parameters
    CONSISTENCY_WEIGHT = 0.5  # Weight for consistency loss
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. NEUTRALIZE CONTENT
# ============================================================================
def neutralize_content(text):
    """
    Remove plausibility cues by replacing content words with neutral symbols.
    This creates a "bias-free" version of the syllogism.
    """
    # Common quantifiers and logical words to preserve
    preserve = {'all', 'some', 'no', 'none', 'every', 'each', 'any',
               'are', 'is', 'not', 'therefore', 'if', 'then',
               'premise', 'conclusion', '1', '2', ':', '.', ','}
    
    words = text.split()
    result = []
    symbol_map = {}
    symbol_idx = 0
    symbols = ['THING_A', 'THING_B', 'THING_C', 'THING_D', 'THING_E']
    
    for word in words:
        word_clean = word.strip('.,!?:;()[]"\'').lower()
        
        if word_clean in preserve or len(word_clean) <= 2 or word_clean.isdigit():
            result.append(word)
        else:
            if word_clean not in symbol_map:
                if symbol_idx < len(symbols):
                    symbol_map[word_clean] = symbols[symbol_idx]
                    symbol_idx += 1
                else:
                    symbol_map[word_clean] = f'THING_{symbol_idx}'
                    symbol_idx += 1
            result.append(symbol_map[word_clean])
    
    return ' '.join(result)

# ============================================================================
# 5. CREATE PROMPTS
# ============================================================================
def parse_syllogism(text):
    premise1_match = re.search(r'Premise 1[:\s]*(.+?)(?=Premise 2|$)', text, re.IGNORECASE | re.DOTALL)
    premise2_match = re.search(r'Premise 2[:\s]*(.+?)(?=Conclusion|$)', text, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if premise1_match and premise2_match and conclusion_match:
        return {
            'premise1': premise1_match.group(1).strip(),
            'premise2': premise2_match.group(1).strip(),
            'conclusion': conclusion_match.group(1).strip()
        }
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= 3:
        return {'premise1': sentences[0], 'premise2': sentences[1], 'conclusion': '. '.join(sentences[2:])}
    return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_prompt(text, neutralize=False):
    if neutralize:
        text = neutralize_content(text)
    
    parsed = parse_syllogism(text)
    return """Analyze logical validity only. Focus on STRUCTURE, ignore content truth.

[P1]: {premise1}
[P2]: {premise2}
[C]: {conclusion}

Logically valid?""".format(**parsed)

# ============================================================================
# 6. BCT DATASET - Returns pairs
# ============================================================================
class BCTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Original prompt
        text_orig = create_prompt(item['syllogism'], neutralize=False)
        enc_orig = self.tokenizer(text_orig, max_length=self.max_length, padding='max_length',
                                  truncation=True, return_tensors='pt')
        
        result = {
            'input_ids': enc_orig['input_ids'].squeeze(),
            'attention_mask': enc_orig['attention_mask'].squeeze(),
        }
        
        if not self.is_test:
            result['labels'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
            
            # Neutralized prompt for consistency training
            text_neutral = create_prompt(item['syllogism'], neutralize=True)
            enc_neutral = self.tokenizer(text_neutral, max_length=self.max_length, padding='max_length',
                                         truncation=True, return_tensors='pt')
            result['input_ids_neutral'] = enc_neutral['input_ids'].squeeze()
            result['attention_mask_neutral'] = enc_neutral['attention_mask'].squeeze()
        
        return result

# ============================================================================
# 7. BCT TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        input_ids_n = batch['input_ids_neutral'].to(device)
        attention_mask_n = batch['attention_mask_neutral'].to(device)
        
        optimizer.zero_grad()
        
        # Forward passes for both versions
        logits_orig = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_neutral = model(input_ids=input_ids_n, attention_mask=attention_mask_n).logits
        
        # Classification loss (original)
        ce_loss = ce_fn(logits_orig, labels)
        
        # Consistency loss: predictions should match between original and neutral
        # Using KL divergence for soft consistency
        probs_orig = F.softmax(logits_orig, dim=-1)
        probs_neutral = F.softmax(logits_neutral, dim=-1)
        consistency_loss = F.kl_div(
            F.log_softmax(logits_orig, dim=-1),
            probs_neutral,
            reduction='batchmean'
        ) + F.kl_div(
            F.log_softmax(logits_neutral, dim=-1),
            probs_orig,
            reduction='batchmean'
        )
        
        loss = ce_loss + Config.CONSISTENCY_WEIGHT * consistency_loss
        
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
# 8. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 18: Bias-Augmented Consistency Training (BCT)")
    print(f"  Consistency weight: {Config.CONSISTENCY_WEIGHT}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Demo neutralization
    print("\n📋 Neutralization example:")
    sample = train_data[0]['syllogism'][:100]
    print(f"   Original:   {sample}...")
    print(f"   Neutralized: {neutralize_content(sample)[:100]}...")
    
    # Split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(
        train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = BCTDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = BCTDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = BCTDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2)
    
    print(f"\n🚀 Training with BCT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2
    ).to(device)
    
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
            model.save_pretrained(os.path.join(Config.OUTPUT_DIR, 'best_model'))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("   Early stopping!")
                break
    
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(Config.OUTPUT_DIR, 'best_model')
    ).to(device)
    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    
    print("\n🔮 Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Predicting"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for pred in preds:
                if idx < len(test_data):
                    predictions.append({'id': test_data[idx]['id'], 'validity': bool(pred)})
                    idx += 1
    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    import shutil
    if os.path.exists(os.path.join(Config.OUTPUT_DIR, 'best_model')):
        shutil.rmtree(os.path.join(Config.OUTPUT_DIR, 'best_model'))
    
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
