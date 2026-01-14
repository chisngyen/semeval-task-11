"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 36 - Seed Optimization (10x)
================================================================================
Description: "The Monte Carlo Approach"
             - Runs the SOTA Exp28b Config (Qwen 14B High Rank) 10 times.
             - Uses 10 different random seeds.
             - Saves PREDICTIONS (Probabilities) for each seed.
             - DELETES model checkpoints immediately to save Disk Space.
             - Performs Final Ensemble (Hard Vote & Soft Vote).
             
Model: Qwen/Qwen2.5-14B-Instruct
Method: QLoRA (Rank 64) + Fixed Augmentation
Ensemble: Majority Voting & Mean Probability
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys
import shutil

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

# ============================================================================
# 2. IMPORTS
# ============================================================================
import json
import os
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig,
    set_seed, 
    get_scheduler
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training,
    PeftModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
class Config:
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    WORKING_DIR = '/kaggle/working'
    OUTPUT_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Model
    MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
    MAX_LENGTH = 512
    
    # QLoRA params (Exp28b - High Rank)
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    
    # Training (Per Seed)
    BATCH_SIZE = 2
    GRAD_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03
    
    # Augmentation
    CF_RATIO = 0.5
    VAL_SPLIT = 0.1
    
    # Seeds to run
    SEEDS = [42, 100, 2024, 7, 123, 888, 999, 10, 555, 3000] 

# ============================================================================
# 4. UTILS & DATA (Copied from Exp28b)
# ============================================================================
class CounterfactualGenerator:
    def __init__(self, all_data):
        self.invalid_conclusions = []
        for item in all_data:
            parsed = self.parse_conclusion(item['syllogism'])
            if not item['validity']:
                self.invalid_conclusions.append(parsed)
    
    def parse_conclusion(self, text):
        match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return match.group(1).strip() if match else (sentences[-1] if sentences else "")
    
    def get_premises(self, text):
        match = re.search(r'^(.+?)(?=Conclusion)', text, re.IGNORECASE | re.DOTALL)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if match: return match.group(1).strip()
        if len(sentences) >= 2: return '. '.join(sentences[:-1]) + '.'
        return text
    
    def create_counterfactual(self, item):
        if not item['validity']: return None 
        premises = self.get_premises(item['syllogism'])
        if self.invalid_conclusions:
            new_conclusion = random.choice(self.invalid_conclusions)
            return {
                'syllogism': f"{premises} Conclusion: {new_conclusion}",
                'validity': False,
                'is_counterfactual': True
            }
        return None

def parse_syllogism(text):
    premise1_match = re.search(r'Premise 1[:\s]*(.+?)(?=Premise 2|$)', text, re.IGNORECASE | re.DOTALL)
    premise2_match = re.search(r'Premise 2[:\s]*(.+?)(?=Conclusion|$)', text, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    if premise1_match and premise2_match and conclusion_match:
        return {'premise1': premise1_match.group(1).strip(), 'premise2': premise2_match.group(1).strip(), 'conclusion': conclusion_match.group(1).strip()}
    return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_prompt(syllogism_text):
    parsed = parse_syllogism(syllogism_text)
    return f"""<|im_start|>system
You are a master logician. Determine if the conclusion strictly follows from the premises.
Structure: Valid (logic follows) or Invalid (logic does not follow).
Ignore real-world plausibility. Focus only on the logical structure.<|im_end|>
<|im_start|>user
[PREMISE 1]: {parsed['premise1']}
[PREMISE 2]: {parsed['premise2']}
[CONCLUSION]: {parsed['conclusion']}
Does the conclusion logically follow?<|im_end|>
<|im_start|>assistant
"""

class QwenAugmentedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False, cf_generator=None, cf_ratio=0.5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.samples = []
        if not is_test and cf_generator:
            for item in data:
                self.samples.append(item)
                if item['validity'] and random.random() < cf_ratio:
                    cf = cf_generator.create_counterfactual(item)
                    if cf: self.samples.append(cf)
        else:
            self.samples = data
            
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = create_prompt(item['syllogism'])
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        result = {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}
        if not self.is_test:
            result['labels'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
        return result

# ============================================================================
# 5. CORE FUNCTIONS
# ============================================================================
def train_one_seed(seed_idx, seed, train_data, test_data, tokenizer, cf_generator):
    print(f"\n🌱 [Seed {seed}] Starting Training (Run {seed_idx+1}/{len(Config.SEEDS)})...")
    set_seed(seed)
    
    # Split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=seed)
    
    # Datasets
    train_dataset = QwenAugmentedDataset(train_split, tokenizer, Config.MAX_LENGTH, cf_generator=cf_generator, cf_ratio=Config.CF_RATIO)
    val_dataset = QwenAugmentedDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = QwenAugmentedDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Model Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False,
        r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA, lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    num_steps = len(train_loader) * Config.NUM_EPOCHS // Config.GRAD_ACCUMULATION
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=int(num_steps * Config.WARMUP_RATIO), num_training_steps=num_steps)
    
    # Training Loop
    best_val_acc = 0
    save_dir = os.path.join(Config.WORKING_DIR, f'temp_model_seed_{seed}')
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)):
            outputs = model(input_ids=batch['input_ids'].to(model.device), 
                          attention_mask=batch['attention_mask'].to(model.device), 
                          labels=batch['labels'].to(model.device))
            loss = outputs.loss / Config.GRAD_ACCUMULATION
            loss.backward()
            batch_loss += loss.item()
            
            if (step + 1) % Config.GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += batch_loss
                batch_loss = 0
        
        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'].to(model.device), 
                              attention_mask=batch['attention_mask'].to(model.device))
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        val_acc = accuracy_score(labels, preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_dir)
            
    print(f"   🏆 Best Val Acc for Seed {seed}: {best_val_acc:.4f}")
    
    # Inference on Test (Load Best)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    
    print("   🔮 Predicting on Test...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, save_dir)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Predicting", leave=False):
            outputs = model(input_ids=batch['input_ids'].to(model.device), 
                          attention_mask=batch['attention_mask'].to(model.device))
            # Save PROBABILITIES for Soft Voting
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy() # [B, 2]
            
            for prob in probs:
                if idx < len(test_data):
                    pred_class = bool(np.argmax(prob))
                    pred_prob_valid = float(prob[1]) # Probability of Valid (Class 1)
                    
                    predictions.append({
                        'id': test_data[idx]['id'], 
                        'validity': pred_class,
                        'prob_valid': pred_prob_valid, # CRITICAL FOR ENSEMBLE
                        'seed': seed,
                        'val_acc': best_val_acc
                    })
                    idx += 1
    
    # CLEANUP
    del model, base_model
    torch.cuda.empty_cache()
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) # DELETE MODEL TO SAVE SPACE
        print("   🧹 Deleted temp model checkpoint.")
        
    return predictions

# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 36: Seed Optimization (10x Qwen 14B High Rank)")
    print("=" * 70)
    
    with open(Config.TRAIN_PATH, 'r') as f: train_data = json.load(f)
    with open(Config.TEST_PATH, 'r') as f: test_data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    cf_generator = CounterfactualGenerator(train_data)
    
    all_runs_predictions = []
    
    # --- RUN LOOP ---
    for i, seed in enumerate(Config.SEEDS):
        preds = train_one_seed(i, seed, train_data, test_data, tokenizer, cf_generator)
        all_runs_predictions.append(preds)
        
        # Save individual run
        out_file = os.path.join(Config.WORKING_DIR, f'pred_seed_{seed}.json')
        with open(out_file, 'w') as f: json.dump(preds, f, indent=2)
        print(f"   💾 Saved predictions to {out_file}")

    # --- ENSEMBLE ---
    print("\n🎹 Performing Advanced Ensembles...")
    
    # 1. Identify Top 3 Seeds (by Val Acc)
    # We need to extract val_acc from the first sample of each run (it's consistent per run)
    run_val_accs = [(i, run[0]['val_acc']) for i, run in enumerate(all_runs_predictions)]
    run_val_accs.sort(key=lambda x: x[1], reverse=True)
    top3_indices = [x[0] for x in run_val_accs[:3]]
    print(f"   🏆 Top 3 Seeds indices: {top3_indices} (Accs: {[x[1] for x in run_val_accs[:3]]})")
    
    # Weights for Weighted Ensemble (Normalize Val Accs)
    val_accs_np = np.array([x[1] for x in run_val_accs])
    # Simple weighting: Raise to power of 10 to emphasize differences in high 90s range
    weights = val_accs_np ** 10 
    weights = weights / np.sum(weights)
    # Map run_idx to weight
    weight_map = {idx: w for idx, w in zip([x[0] for x in run_val_accs], weights)}

    # Map ID -> List of (validity, prob)
    sample_ids = [p['id'] for p in all_runs_predictions[0]]
    
    ens_majority = []
    ens_mean = []
    ens_max = []
    ens_weighted = []
    ens_top3 = []
    
    for i, sid in enumerate(sample_ids):
        # Gather data for this sample across all runs
        probs = [run[i]['prob_valid'] for run in all_runs_predictions]
        votes = [run[i]['validity'] for run in all_runs_predictions]
        
        # 1. Majority Vote (Hard)
        vote_count = Counter(votes)
        majority_pred = vote_count.most_common(1)[0][0]
        ens_majority.append({'id': sid, 'validity': bool(majority_pred)})
        
        # 2. Mean Probability (Soft)
        mean_prob = np.mean(probs)
        ens_mean.append({'id': sid, 'validity': bool(mean_prob > 0.5)})
        
        # 3. Max Probability (Confidence)
        # If any model is very confident (near 1.0 or 0.0), let it win
        # We look at deviation from 0.5
        deviations = [abs(p - 0.5) for p in probs]
        max_conf_idx = np.argmax(deviations)
        max_prob_pred = probs[max_conf_idx] > 0.5
        ens_max.append({'id': sid, 'validity': bool(max_prob_pred)})
        
        # 4. Weighted Probability
        weighted_prob = sum(p * weight_map[r_idx] for r_idx, p in enumerate(probs))
        ens_weighted.append({'id': sid, 'validity': bool(weighted_prob > 0.5)})
        
        # 5. Top 3 Soft Vote
        top3_probs = [all_runs_predictions[idx][i]['prob_valid'] for idx in top3_indices]
        top3_mean = np.mean(top3_probs)
        ens_top3.append({'id': sid, 'validity': bool(top3_mean > 0.5)})

    # Save All Variations
    files_to_zip = []
    
    variations = [
        ('ensemble_majority.json', ens_majority),
        ('ensemble_mean_prob.json', ens_mean),
        ('ensemble_max_conf.json', ens_max),
        ('ensemble_weighted.json', ens_weighted),
        ('ensemble_top3.json', ens_top3)
    ]
    
    for fname, data in variations:
        path = os.path.join(Config.WORKING_DIR, fname)
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        files_to_zip.append(path)
        print(f"   💾 Saved {fname}")
        
    # Zip everything (Ensembles + Individual Predictions)
    # Add individual prediction files to zip list (already created in loop)
    # We find them by pattern
    for f in os.listdir(Config.WORKING_DIR):
        if f.startswith('pred_seed_') and f.endswith('.json'):
            files_to_zip.append(os.path.join(Config.WORKING_DIR, f))
            
    import zipfile
    with zipfile.ZipFile(Config.OUTPUT_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))
        
    print(f"\n✅ All Done! ZIP contains {len(files_to_zip)} files: {Config.OUTPUT_ZIP_PATH}")
    print("   Includes: Majority, Mean, Max, Weighted, Top3, and 10 individual seeds.")

if __name__ == '__main__':
    main()
