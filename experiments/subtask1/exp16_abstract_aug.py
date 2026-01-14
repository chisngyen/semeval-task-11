"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 16 - Exp12 + Abstract Augmentation
================================================================================
Description: Combine Exp12 (Logic-Aware Prompt, ACC=96.34%) with:
             - Data augmentation using ABSTRACT SYMBOLS
             - Train on BOTH real content AND abstract versions
             - This helps model learn form-based reasoning, not content
             
Research: "SFT with abstract/pseudo-word formulas eliminates content bias"
          (ACL 2024)
             
Goal: Keep high accuracy (~96%) but reduce TCE (<3.0)
             
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
    OUTPUT_DIR = '/kaggle/working/deberta_exp16'
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
    
    # Abstract augmentation ratio
    ABSTRACT_RATIO = 0.5  # 50% original, 50% abstract
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. ABSTRACT SYMBOL GENERATOR
# ============================================================================
class AbstractSymbolGenerator:
    """
    Convert content words to abstract symbols.
    This helps model focus on logical FORM, not content.
    
    Example:
    "All dogs are mammals" -> "All X are Y"
    "All mammals breathe" -> "All Y breathe"  
    -> Model learns the logical structure: All X are Y, All Y are Z ∴ All X are Z
    """
    
    SYMBOLS = ['X', 'Y', 'Z', 'W', 'V', 'P', 'Q', 'R', 'S', 'T']
    
    def __init__(self):
        self.word_to_symbol = {}
        self.used_symbols = 0
    
    def reset(self):
        self.word_to_symbol = {}
        self.used_symbols = 0
    
    def get_symbol(self, word):
        word_lower = word.lower().strip()
        if word_lower not in self.word_to_symbol:
            if self.used_symbols < len(self.SYMBOLS):
                self.word_to_symbol[word_lower] = self.SYMBOLS[self.used_symbols]
                self.used_symbols += 1
            else:
                # If we run out of symbols, use numbered ones
                self.word_to_symbol[word_lower] = f"T{self.used_symbols}"
                self.used_symbols += 1
        return self.word_to_symbol[word_lower]
    
    def abstractify(self, text):
        """Convert content words to abstract symbols."""
        self.reset()
        
        # Find noun phrases (simplified: just nouns and adjective+nouns)
        # This is a simplified approach - could be improved with NLP
        words = text.split()
        result = []
        
        # Common logical words to preserve
        preserve = {'all', 'some', 'no', 'none', 'are', 'is', 'therefore', 
                   'if', 'then', 'not', 'every', 'each', 'any', 'the', 'a', 'an',
                   'premise', 'conclusion', 'valid', 'invalid', '1', '2', ':', '.', ',',
                   'does', 'follow', 'from', 'logically', 'necessarily'}
        
        for word in words:
            word_clean = word.strip('.,!?:;()[]"\'').lower()
            
            if word_clean in preserve or len(word_clean) <= 2:
                result.append(word)
            elif word_clean.isdigit():
                result.append(word)
            else:
                # Replace with abstract symbol
                symbol = self.get_symbol(word_clean)
                # Preserve original capitalization pattern
                if word[0].isupper():
                    result.append(symbol)
                else:
                    result.append(symbol.lower())
        
        return ' '.join(result)

# ============================================================================
# 5. SYLLOGISM PARSER (FROM EXP12)
# ============================================================================
def parse_syllogism(syllogism_text):
    text = syllogism_text.strip()
    
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
    else:
        return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_logic_aware_prompt(syllogism_text, use_abstract=False):
    """Create structured prompt, optionally with abstract symbols."""
    parsed = parse_syllogism(syllogism_text)
    
    if use_abstract:
        generator = AbstractSymbolGenerator()
        # Apply abstraction to each part
        text_combined = f"{parsed['premise1']} {parsed['premise2']} {parsed['conclusion']}"
        abstract_text = generator.abstractify(text_combined)
        
        # Re-parse the abstract version
        abstract_sentences = abstract_text.split('.')
        abstract_sentences = [s.strip() for s in abstract_sentences if s.strip()]
        
        if len(abstract_sentences) >= 3:
            parsed = {
                'premise1': abstract_sentences[0],
                'premise2': abstract_sentences[1],
                'conclusion': '. '.join(abstract_sentences[2:])
            }
        elif len(abstract_sentences) >= 1:
            parsed = {
                'premise1': abstract_sentences[0],
                'premise2': abstract_sentences[1] if len(abstract_sentences) > 1 else '',
                'conclusion': abstract_sentences[-1] if len(abstract_sentences) > 2 else ''
            }
    
    prompt = """Analyze the logical validity of this syllogism. Focus ONLY on logical structure, NOT on whether the content is true in the real world.

=== SYLLOGISM STRUCTURE ===
[PREMISE 1]: {premise1}
[PREMISE 2]: {premise2}
[CONCLUSION]: {conclusion}

=== LOGICAL ANALYSIS ===
A syllogism is VALID if the conclusion necessarily follows from the premises, regardless of whether the premises are actually true.

Question: Does the conclusion logically follow from the premises?
Answer (valid/invalid):""".format(
        premise1=parsed['premise1'],
        premise2=parsed['premise2'],
        conclusion=parsed['conclusion']
    )
    
    return prompt

# ============================================================================
# 6. AUGMENTED DATASET
# ============================================================================
class AugmentedDataset(Dataset):
    """
    Dataset that includes BOTH original and abstract versions.
    This teaches model to reason about form, not content.
    """
    def __init__(self, data, tokenizer, max_length, is_test=False, abstract_ratio=0.5):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.abstract_ratio = abstract_ratio
        
        if not is_test:
            # Create augmented training data
            self.augmented_data = []
            for item in data:
                # Original version
                self.augmented_data.append({
                    'syllogism': item['syllogism'],
                    'validity': item['validity'],
                    'use_abstract': False
                })
                # Abstract version (only for training)
                if random.random() < abstract_ratio:
                    self.augmented_data.append({
                        'syllogism': item['syllogism'],
                        'validity': item['validity'],
                        'use_abstract': True
                    })
            print(f"   Augmented dataset: {len(self.augmented_data)} samples (original + abstract)")
        else:
            self.augmented_data = [{'syllogism': item['syllogism'], 'use_abstract': False} for item in data]
    
    def __len__(self):
        return len(self.augmented_data)
    
    def __getitem__(self, idx):
        item = self.augmented_data[idx]
        text = create_logic_aware_prompt(item['syllogism'], use_abstract=item.get('use_abstract', False))
        
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
# 7. TRAINING LOOP
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
    print("Exp 16: Exp12 + Abstract Symbol Augmentation")
    print("  - Logic-Aware Prompt (from Exp12)")
    print("  - Abstract symbol data augmentation")
    print("  - Goal: High ACC + Low TCE")
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
    
    # Demo abstract conversion
    print("\n📋 Abstract conversion example:")
    sample = train_data[0]['syllogism']
    print(f"   Original: {sample[:80]}...")
    generator = AbstractSymbolGenerator()
    abstract = generator.abstractify(sample)
    print(f"   Abstract: {abstract[:80]}...")
    
    # Split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(
        train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED
    )
    
    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = AugmentedDataset(train_split, tokenizer, Config.MAX_LENGTH, abstract_ratio=Config.ABSTRACT_RATIO)
    val_dataset = AugmentedDataset(val_split, tokenizer, Config.MAX_LENGTH, abstract_ratio=0)  # No augmentation for val
    test_dataset = AugmentedDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2)
    
    # Model
    print(f"\n🚀 Training with Abstract Augmentation...")
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
    
    # Load best model
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(Config.OUTPUT_DIR, 'best_model')
    ).to(device)
    print(f"\n✅ Best Val Accuracy: {best_acc:.4f}")
    
    # Inference (always use original content for test)
    print("\n🔮 Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
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
    
    # Cleanup
    import shutil
    if os.path.exists(os.path.join(Config.OUTPUT_DIR, 'best_model')):
        shutil.rmtree(os.path.join(Config.OUTPUT_DIR, 'best_model'))
    
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
