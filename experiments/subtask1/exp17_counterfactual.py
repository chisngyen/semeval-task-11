"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 17 - Counterfactual Data Augmentation
================================================================================
Description: Create counterfactual training pairs by swapping conclusions
             between valid/invalid syllogisms while keeping premises.
             
Research: "Counterfactual Data Augmentation improves model robustness"
          (ACL/NeurIPS 2024)
             
Idea: For each syllogism, create a counterfactual version:
      - If valid: create invalid version by random conclusion
      - If invalid: create valid version with correct conclusion
      This teaches model to focus on LOGICAL RELATION, not content
             
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
    OUTPUT_DIR = '/kaggle/working/deberta_exp17'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 384
    
    BATCH_SIZE = 12
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    LABEL_SMOOTHING = 0.1
    
    # Counterfactual augmentation
    CF_RATIO = 0.3  # 30% counterfactual examples
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. COUNTERFACTUAL GENERATOR
# ============================================================================
class CounterfactualGenerator:
    """
    Generate counterfactual syllogisms by swapping conclusions.
    This creates minimal pairs that differ only in logical validity.
    """
    
    def __init__(self, all_data):
        # Group by validity
        self.valid_conclusions = []
        self.invalid_conclusions = []
        
        for item in all_data:
            parsed = self.parse_conclusion(item['syllogism'])
            if item['validity']:
                self.valid_conclusions.append(parsed)
            else:
                self.invalid_conclusions.append(parsed)
        
        print(f"   Counterfactual pool: {len(self.valid_conclusions)} valid, {len(self.invalid_conclusions)} invalid conclusions")
    
    def parse_conclusion(self, text):
        """Extract conclusion from syllogism."""
        match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            return sentences[-1]
        return ""
    
    def get_premises(self, text):
        """Extract premises from syllogism."""
        # Try to get everything before conclusion
        match = re.search(r'^(.+?)(?=Conclusion)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            return '. '.join(sentences[:-1]) + '.'
        return text
    
    def create_counterfactual(self, item):
        """
        Create counterfactual by swapping conclusion with opposite validity.
        """
        premises = self.get_premises(item['syllogism'])
        
        if item['validity']:
            # Valid -> Invalid: use random invalid conclusion
            if self.invalid_conclusions:
                new_conclusion = random.choice(self.invalid_conclusions)
                new_validity = False
            else:
                return None
        else:
            # Invalid -> Valid: use random valid conclusion  
            if self.valid_conclusions:
                new_conclusion = random.choice(self.valid_conclusions)
                new_validity = True
            else:
                return None
        
        # Construct new syllogism
        new_syllogism = f"{premises} Conclusion: {new_conclusion}"
        
        return {
            'syllogism': new_syllogism,
            'validity': new_validity,
            'is_counterfactual': True
        }

# ============================================================================
# 5. SYLLOGISM PARSER (FROM EXP12)
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
    elif len(sentences) == 2:
        return {'premise1': sentences[0], 'premise2': '', 'conclusion': sentences[1]}
    return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_logic_prompt(syllogism_text):
    parsed = parse_syllogism(syllogism_text)
    return """Analyze logical validity. Focus on STRUCTURE, not content truth.

[PREMISE 1]: {premise1}
[PREMISE 2]: {premise2}
[CONCLUSION]: {conclusion}

Does the conclusion logically follow from the premises?""".format(**parsed)

# ============================================================================
# 6. AUGMENTED DATASET
# ============================================================================
class CounterfactualDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False, cf_generator=None, cf_ratio=0.3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        if not is_test and cf_generator:
            # Create augmented training data
            self.samples = []
            for item in data:
                # Original
                self.samples.append({
                    'syllogism': item['syllogism'],
                    'validity': item['validity']
                })
                # Counterfactual
                if random.random() < cf_ratio:
                    cf = cf_generator.create_counterfactual(item)
                    if cf:
                        self.samples.append(cf)
            print(f"   Dataset: {len(self.samples)} samples (original + counterfactual)")
        else:
            self.samples = [{'syllogism': item['syllogism']} for item in data]
            if not is_test:
                for i, item in enumerate(data):
                    self.samples[i]['validity'] = item['validity']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = create_logic_prompt(item['syllogism'])
        
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
# 7. TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    ce_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = ce_fn(outputs.logits, labels)
        
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 17: Counterfactual Data Augmentation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Create counterfactual generator
    cf_generator = CounterfactualGenerator(train_data)
    
    # Split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(
        train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = CounterfactualDataset(train_split, tokenizer, Config.MAX_LENGTH, 
                                          cf_generator=cf_generator, cf_ratio=Config.CF_RATIO)
    val_dataset = CounterfactualDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = CounterfactualDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2)
    
    print(f"\n🚀 Training with Counterfactual Augmentation...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, problem_type="single_label_classification"
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
