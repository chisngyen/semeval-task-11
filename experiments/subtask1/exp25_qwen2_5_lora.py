"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 25 - Qwen 2.5 7B (LoRA)
================================================================================
Description: Fine-tuning Qwen 2.5 7B Instruct (SOTA open-weights for logic)
             using QLoRA (4-bit quantization + LoRA).
             
             Qwen 2.5 is known as a "Math Powerhouse" and should excel at 
             formal logic better than encoder-only models.
             
Model: Qwen/Qwen2.5-7B-Instruct
Method: QLoRA (4-bit, rank 16)
Prompt: Logic-Aware Prompt (Exp12)
             
Difficulty: Very Hard (Requires GPU memory management)
Hardware: Kaggle H100 GPU (Recommended) or 2xT4
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys

def install_packages():
    # Install specific versions for QLoRA compatibility
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
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig,
    set_seed, 
    get_scheduler,
    DataCollatorWithPadding
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
    OUTPUT_DIR = '/kaggle/working/qwen_exp25'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    MAX_LENGTH = 512  # LLMs can handle longer context
    
    # QLoRA params
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # Training params
    BATCH_SIZE = 4      # Smaller batch size for 7B model
    GRAD_ACCUMULATION = 4
    LEARNING_RATE = 2e-4 # LoRA usually needs higher LR
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. LOGIC-AWARE PROMPT (From Exp12)
# ============================================================================
import re
def parse_syllogism(text):
    premise1_match = re.search(r'Premise 1[:\s]*(.+?)(?=Premise 2|$)', text, re.IGNORECASE | re.DOTALL)
    premise2_match = re.search(r'Premise 2[:\s]*(.+?)(?=Conclusion|$)', text, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    if premise1_match and premise2_match and conclusion_match:
        return {'premise1': premise1_match.group(1).strip(), 'premise2': premise2_match.group(1).strip(), 'conclusion': conclusion_match.group(1).strip()}
    return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_prompt(syllogism_text):
    parsed = parse_syllogism(syllogism_text)
    # Using specific Qwen/Instruct format can help, but for classification 
    # we just need the model to output checks on the [CLS] token equivalent.
    # We will wrap it in an instruction format for semantic clarity.
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
# 5. DATASET
# ============================================================================
class QwenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = create_prompt(item['syllogism'])
        
        # Tokenize (Need to explicitly set padding side for some LLMs, but Qwen usually auto-configures)
        # Note: AutoModelForSequenceClassification uses the last token (usually EOS) for prediction in some libs,
        # or the first [CLS]. Qwen uses the last non-pad token typically.
        # We'll rely on the padding token.
        
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
# 6. TRAINING LOOP
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
# 7. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(f"Exp 25: {Config.MODEL_NAME} (QLoRA)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Data
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(train_data, test_size=Config.VAL_SPLIT, stratify=strat_keys, random_state=Config.SEED)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Datasets
    train_dataset = QwenDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = QwenDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = QwenDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n🚀 Loading Model with QLoRA...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # Target modules for Qwen2 (usually q_proj, k_proj, v_proj, o_proj, down_proj, up_proj, gate_proj)
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
    
    # Pad token id fix for classification
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
        print(f"   Train Loss: {train_loss:.4f}")
        
        # Validation
        val_acc = evaluate(model, val_loader, device)
        print(f"   Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"   New Best! Saving adapter...")
            model.save_pretrained(os.path.join(Config.OUTPUT_DIR, 'best_model'))
            
    print(f"\n✅ Best Validation Accuracy: {best_acc:.4f}")
    
    # Inference
    print("\n🔮 Generating Predictions...")
    # Reload best adapter
    del model
    torch.cuda.empty_cache()
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, 
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Fix: Ensure padding token is set for inference (critical for batching)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # The peft library automatically injects adapters
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
