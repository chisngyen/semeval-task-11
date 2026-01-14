"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 31 - Test-Time Augmentation (TTA)
================================================================================
Description: Improves inference robustness by checking logical consistency.
             
             Method:
             1. Load SOTA Model (Exp28b - Qwen 14B High Rank).
             2. For each test sample:
                - Prediction A: Original Syllogism
                - Prediction B: Swapped Premises (Logically equivalent)
             3. Combine predictions (Union/Voting).
             
             Hypothesis:
             Valid syllogisms must remain valid regardless of premise order.
             If model disagrees with itself, we can rely on the more "confident" 
             prediction or default to a safe class.
             
Model: Uses trained adapters from Exp28b
================================================================================
"""

# ============================================================================
# 0. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys

def install_packages():
    packages = [
        'transformers>=4.38.0', 
        'peft>=0.10.0', 
        'bitsandbytes>=0.43.0', 
        'accelerate>=0.28.0',
        'scikit-learn',
        'protobuf==3.20.3'
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")

install_packages()

import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig
)
from peft import PeftModel
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    # Point to the SOTA model (Exp28b - Loaded from Dataset)
    BASE_MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
    ADAPTER_PATH = '/kaggle/input/qwen2-5/qwen14b_exp28b_high_rank/best_model'
    
    OUTPUT_FILE = '/kaggle/working/predictions.json'
    ZIP_FILE = '/kaggle/working/predictions.zip'
    
    MAX_LENGTH = 512
    BATCH_SIZE = 4 # Higher batch size for inference

# ============================================================================
# LOGIC PARSING & TTA
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
    return None

def create_prompt(p1, p2, conc):
    return f"""<|im_start|>system
You are a master logician. Determine if the conclusion strictly follows from the premises.
Structure: Valid (logic follows) or Invalid (logic does not follow).
Ignore real-world plausibility. Focus only on the logical structure.<|im_end|>
<|im_start|>user
[PREMISE 1]: {p1}
[PREMISE 2]: {p2}
[CONCLUSION]: {conc}
Does the conclusion logically follow?<|im_end|>
<|im_start|>assistant
"""

# ============================================================================
# DATASET
# ============================================================================
class TTADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        parsed = parse_syllogism(item['syllogism'])
        
        inputs = []
        
        # Original
        if parsed:
            p1, p2, c = parsed['premise1'], parsed['premise2'], parsed['conclusion']
            # Variation 1: Original
            text1 = create_prompt(p1, p2, c)
            # Variation 2: Swapped Premises (Logically Equivalent)
            text2 = create_prompt(p2, p1, c)
        else:
            # Fallback for parsing failure
            text1 = item['syllogism']
            text2 = item['syllogism']

        encodings = []
        for text in [text1, text2]:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze()
            })
            
        return item['id'], encodings[0], encodings[1]

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("Exp 31: Test-Time Augmentation (TTA)")
    print("Target Model: Exp28b Adapters")
    print("="*60)
    
    # 1. Load Data
    with open(Config.TEST_PATH, 'r') as f:
        test_data = json.load(f)
        
    # 2. Load Model
    print("Loading Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load Exp28b Adapters
    if os.path.exists(Config.ADAPTER_PATH):
        print(f"Loading Adapters from: {Config.ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, Config.ADAPTER_PATH)
    else:
        print("⚠️ ADAPTER NOT FOUND! Checking local directory...")
        # Fallback to current directory if script ran in same session
        model = PeftModel.from_pretrained(base_model, '/kaggle/working/qwen14b_exp28b_high_rank/best_model')
        
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Predict with TTA
    dataset = TTADataset(test_data, tokenizer, Config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    results = []
    
    print("Running Inference (Original + Swapped)...")
    with torch.no_grad():
        for batch_ids, batch_orig, batch_swap in tqdm(dataloader):
            # Pass 1: Original
            out1 = model(
                input_ids=batch_orig['input_ids'].to(device),
                attention_mask=batch_orig['attention_mask'].to(device)
            )
            prob1 = torch.softmax(out1.logits, dim=1)[:, 1] # Prob of Valid (1)
            
            # Pass 2: Swapped
            out2 = model(
                input_ids=batch_swap['input_ids'].to(device),
                attention_mask=batch_swap['attention_mask'].to(device)
            )
            prob2 = torch.softmax(out2.logits, dim=1)[:, 1]
            
            # Combine: Average Probability
            final_probs = (prob1 + prob2) / 2.0
            preds = (final_probs > 0.5).long().cpu().numpy()
            
            for i, _id in enumerate(batch_ids):
                results.append({'id': _id, 'validity': bool(preds[i])})
                
    # 4. Save
    with open(Config.OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    import zipfile
    with zipfile.ZipFile(Config.ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.OUTPUT_FILE, 'predictions.json')
        
    print(f"✅ Saved TTA Results: {Config.ZIP_FILE}")

if __name__ == '__main__':
    main()
