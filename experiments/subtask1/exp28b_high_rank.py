"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 28b - Qwen 14B High Rank
================================================================================
Description: Variation of Exp28 with Higher LoRA Rank.
             
             Changes from Exp28:
             - LORA_R: 16 -> 64
             - LORA_ALPHA: 32 -> 128
             
             Hypothesis: 
             - Higher rank allows the model to learn more complex patterns 
               (better fitting) given the large amount of data/augmentation.
             - H100 has enough memory to handle this increase.
             
Model: Qwen/Qwen2.5-14B-Instruct
Method: QLoRA (4-bit) + Fixed Counterfactual Augmentation
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
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
    prepare_model_for_kbit_training
)
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
    OUTPUT_DIR = '/kaggle/working/qwen14b_exp28b_high_rank'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # 14B Model
    MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
    MAX_LENGTH = 512
    
    # QLoRA params (HIGH RANK)
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    
    # Training params
    BATCH_SIZE = 2
    GRAD_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03
    
    # Augmentation
    CF_RATIO = 0.5
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. COUNTERFACTUAL GENERATOR
# ============================================================================
class CounterfactualGenerator:
    def __init__(self, all_data):
        self.invalid_conclusions = []
        for item in all_data:
            parsed = self.parse_conclusion(item['syllogism'])
            if not item['validity']:
                self.invalid_conclusions.append(parsed)
        print(f"   CF pool: {len(self.invalid_conclusions)} invalid conclusions")
    
    def parse_conclusion(self, text):
        match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""
    
    def get_premises(self, text):
        match = re.search(r'^(.+?)(?=Conclusion)', text, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2: return '. '.join(sentences[:-1]) + '.'
        return text
    
    def create_counterfactual(self, item):
        if not item['validity']: return None 
        
        premises = self.get_premises(item['syllogism'])
        if self.invalid_conclusions:
            new_conclusion = random.choice(self.invalid_conclusions)
            new_syllogism = f"{premises} Conclusion: {new_conclusion}"
            
            return {
                'syllogism': new_syllogism,
                'validity': False,
                'is_counterfactual': True
            }
        return None

# ============================================================================
# 5. LOGIC-AWARE PROMPT
# ============================================================================
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

# ============================================================================
# 6. DATASET WITH AUGMENTATION
# ============================================================================
class QwenAugmentedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False, cf_generator=None, cf_ratio=0.5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        if not is_test and cf_generator:
            self.samples = []
            for item in data:
                self.samples.append(item)
                if item['validity'] and random.random() < cf_ratio:
                    cf = cf_generator.create_counterfactual(item)
                    if cf: self.samples.append(cf)
            print(f"   Dataset: {len(self.samples)} samples (Augmented)")
        else:
            self.samples = data
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        text = create_prompt(item['syllogism'])
        
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
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
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    batch_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        batch_loss += loss.item()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += batch_loss
            batch_loss = 0
            progress_bar.set_postfix({'loss': total_loss / (step // gradient_accumulation_steps + 1)})

    return total_loss / (len(dataloader) / gradient_accumulation_steps)

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
    print(f"Exp 28b: {Config.MODEL_NAME} (14B) - HIGH RANK")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Init Augmentation
    cf_generator = CounterfactualGenerator(train_data)
        
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Datasets (Augmented Train)
    train_dataset = QwenAugmentedDataset(train_split, tokenizer, Config.MAX_LENGTH, cf_generator=cf_generator, cf_ratio=Config.CF_RATIO)
    val_dataset = QwenAugmentedDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = QwenAugmentedDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n🚀 Loading Qwen 14B with QLoRA...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.pad_token_id
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    num_steps = len(train_loader) * Config.NUM_EPOCHS // Config.GRAD_ACCUMULATION
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=int(num_steps * Config.WARMUP_RATIO), num_training_steps=num_steps)
    
    best_acc = 0
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("\n⚔️ Start Training...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, Config.GRAD_ACCUMULATION)
        val_acc = evaluate(model, val_loader, device)
        print(f"   Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(os.path.join(Config.OUTPUT_DIR, 'best_model'))
            
    print(f"\n✅ Best Validation Accuracy: {best_acc:.4f}")
    
    print("\n🔮 Generating Predictions...")
    del model
    torch.cuda.empty_cache()
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    # Fix: Ensure padding token is set for inference (critical for batching)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, os.path.join(Config.OUTPUT_DIR, 'best_model'))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            for pred in preds:
                if idx < len(test_data):
                    predictions.append({'id': test_data[idx]['id'], 'validity': bool(pred)})
                    idx += 1
                    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
        
    print(f"\n✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
