"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 5 - Better Hyperparameters
================================================================================
Description: DeBERTa-v3-large with optimized hyperparameters:
             - More epochs (10)
             - Lower learning rate with cosine schedule
             - Label smoothing
             - Longer warmup
             
Difficulty: Easy (just hyperparameter tuning)
Hardware: Kaggle H100 GPU
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES
# ============================================================================
import subprocess
import sys

def install_packages():
    """Install required packages with compatible versions for Kaggle."""
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
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION - OPTIMIZED HYPERPARAMETERS
# ============================================================================
class Config:
    # Paths
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_exp5'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Model
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    # OPTIMIZED Training Parameters
    BATCH_SIZE = 16  # Smaller for stability
    GRADIENT_ACCUMULATION = 2  # Effective batch = 32
    LEARNING_RATE = 5e-6  # Lower LR
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10  # More epochs
    WARMUP_RATIO = 0.15  # Longer warmup
    LABEL_SMOOTHING = 0.1  # Regularization
    
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. DATASET
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
# 5. METRICS
# ============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Exp 5: DeBERTa-v3-large with Better Hyperparameters")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Stratified split
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    train_split, val_split = train_test_split(
        train_data, test_size=Config.VAL_SPLIT, 
        stratify=strat_keys, random_state=Config.SEED
    )
    print(f"   Train split: {len(train_split)}, Val split: {len(val_split)}")
    
    # Tokenizer
    print(f"\n📥 Loading: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Datasets
    train_dataset = SyllogismDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = SyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = SyllogismDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    # Model
    print("\n🚀 Starting training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Training arguments with LABEL SMOOTHING
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        lr_scheduler_type='cosine',  # Cosine schedule
        label_smoothing_factor=Config.LABEL_SMOOTHING,  # Label smoothing
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=25,
        fp16=True,
        dataloader_num_workers=4,
        report_to='none',
        seed=Config.SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # Cleanup
    import shutil
    if os.path.exists(Config.OUTPUT_DIR):
        for item in os.listdir(Config.OUTPUT_DIR):
            item_path = os.path.join(Config.OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint'):
                shutil.rmtree(item_path)
    
    print("✅ Training completed!")
    
    # Inference
    print("\n🔮 Generating predictions...")
    test_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(test_output.predictions, axis=1)
    
    predictions = [{'id': item['id'], 'validity': bool(predicted_labels[i])} 
                   for i, item in enumerate(test_data)]
    
    # Save
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    print(f"✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print(f"   Valid: {sum(1 for p in predictions if p['validity'])}")
    print(f"   Invalid: {sum(1 for p in predictions if not p['validity'])}")
    
    print("\n" + "=" * 70)
    print("🎉 Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
