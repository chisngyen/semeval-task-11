"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 6 - R-Drop Regularization
================================================================================
Description: DeBERTa-v3-large with R-Drop regularization.
             R-Drop: Run input twice through model, minimize KL-divergence
             between two outputs for consistency regularization.
             
Paper: "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021)
Difficulty: Medium
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
# 3. CONFIGURATION
# ============================================================================
class Config:
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_exp6'
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 8e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 8
    WARMUP_RATIO = 0.1
    
    # R-Drop parameters
    RDROP_ALPHA = 0.7  # Weight for KL loss
    
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
# 5. R-DROP TRAINER
# ============================================================================
class RDropTrainer(Trainer):
    """Custom Trainer with R-Drop regularization."""
    
    def __init__(self, rdrop_alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha
    
    def compute_kl_loss(self, p, q):
        """Compute KL divergence between two distributions."""
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1),
            F.softmax(q, dim=-1),
            reduction='batchmean'
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1),
            F.softmax(p, dim=-1),
            reduction='batchmean'
        )
        return (p_loss + q_loss) / 2
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to add R-Drop regularization."""
        labels = inputs.pop("labels", None)
        
        # First forward pass
        outputs1 = model(**inputs)
        logits1 = outputs1.logits
        
        # Second forward pass (different dropout)
        outputs2 = model(**inputs)
        logits2 = outputs2.logits
        
        # Cross-entropy loss (average of both passes)
        if labels is not None:
            ce_loss1 = F.cross_entropy(logits1, labels)
            ce_loss2 = F.cross_entropy(logits2, labels)
            ce_loss = (ce_loss1 + ce_loss2) / 2
            
            # KL divergence loss
            kl_loss = self.compute_kl_loss(logits1, logits2)
            
            # Total loss
            loss = ce_loss + self.rdrop_alpha * kl_loss
        else:
            loss = None
        
        # Return average logits for evaluation
        outputs1.logits = (logits1 + logits2) / 2
        
        return (loss, outputs1) if return_outputs else loss

# ============================================================================
# 6. MAIN
# ============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    print("=" * 70)
    print("Exp 6: DeBERTa-v3-large with R-Drop Regularization")
    print("=" * 70)
    
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
    
    # Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_dataset = SyllogismDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = SyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = SyllogismDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    # Model
    print("\n🚀 Training with R-Drop (alpha={})...".format(Config.RDROP_ALPHA))
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=2, problem_type="single_label_classification"
    )
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=25,
        fp16=True,
        report_to='none',
        seed=Config.SEED,
    )
    
    # Use R-Drop Trainer
    trainer = RDropTrainer(
        rdrop_alpha=Config.RDROP_ALPHA,
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
    
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    print(f"✅ Saved: {Config.SUBMISSION_ZIP_PATH}")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
