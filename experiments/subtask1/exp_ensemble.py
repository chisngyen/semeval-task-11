"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 4 - Ensemble Methods
================================================================================
Description: Combine predictions from multiple models using various ensemble
             strategies. Inspired by NeuBAROCO's analysis of model biases.
             
Method: Weighted voting, majority voting, or meta-learning
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
    # Pin protobuf to avoid MessageFactory error
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'protobuf==3.20.3'])
    
    packages = [
        'transformers>=4.36.0',
        'accelerate>=0.25.0',
    ]
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
class Config:
    # Paths - Kaggle environment
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/ensemble_models'
    
    # Competition output format
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Ensemble models
    MODELS = [
        'microsoft/deberta-v3-large',
        'microsoft/deberta-v3-base',
        'roberta-large',
    ]
    
    MAX_LENGTH = 256
    
    # Training - H100 optimized
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 4
    
    # Ensemble
    N_FOLDS = 5
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. DATASET CLASS
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
# 5. TRAINING UTILITIES
# ============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def train_single_model(model_name, train_data, val_data, fold_idx, output_dir):
    """Train a single model on given data."""
    print(f"\n   Training {model_name.split('/')[-1]} (Fold {fold_idx + 1})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, problem_type="single_label_classification"
    )
    
    train_dataset = SyllogismDataset(train_data, tokenizer, Config.MAX_LENGTH)
    val_dataset = SyllogismDataset(val_data, tokenizer, Config.MAX_LENGTH)
    
    model_output_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}_fold{fold_idx}")
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,  # Only keep 1 checkpoint
        save_only_model=True,  # Don't save optimizer state
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=100,
        fp16=True,
        report_to='none',
        seed=Config.SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Clean up checkpoints to save disk space
    import shutil
    if os.path.exists(model_output_dir):
        for item in os.listdir(model_output_dir):
            item_path = os.path.join(model_output_dir, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint'):
                shutil.rmtree(item_path)
    
    return trainer, tokenizer

def get_predictions(trainer, tokenizer, test_data):
    """Get predictions from a trained model."""
    test_dataset = SyllogismDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    output = trainer.predict(test_dataset)
    
    # Return probabilities, not just labels
    probs = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
    return probs

# ============================================================================
# 6. ENSEMBLE STRATEGIES
# ============================================================================
def majority_voting(all_predictions):
    """Simple majority voting across all models."""
    # all_predictions: list of [n_samples, 2] probability arrays
    stacked = np.stack(all_predictions)  # [n_models, n_samples, 2]
    votes = np.argmax(stacked, axis=2)  # [n_models, n_samples]
    final = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), 0, votes)
    return final

def weighted_average(all_predictions, weights=None):
    """Weighted average of probabilities."""
    if weights is None:
        weights = np.ones(len(all_predictions)) / len(all_predictions)
    
    weighted_probs = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        weighted_probs += weight * pred
    
    return np.argmax(weighted_probs, axis=1)

def confidence_weighted(all_predictions):
    """Weight by model confidence (entropy-based)."""
    weights = []
    for pred in all_predictions:
        entropy = -np.sum(pred * np.log(pred + 1e-10), axis=1)
        avg_entropy = np.mean(entropy)
        weights.append(1.0 / (avg_entropy + 0.1))
    
    weights = np.array(weights) / sum(weights)
    return weighted_average(all_predictions, weights), weights

# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("SemEval-2026 Task 11 - Subtask 1: Ensemble Methods")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
    
    # K-Fold cross-validation for ensemble
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    labels = [item['validity'] for item in train_data]
    
    all_test_predictions = []
    
    # Train each model with 1 fold (simplified for speed)
    print("\n🚀 Training ensemble models...")
    
    # Split for validation
    train_split, val_split = train_test_split(
        train_data, test_size=0.1, stratify=strat_keys, random_state=Config.SEED
    )
    
    for model_name in Config.MODELS:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        trainer, tokenizer = train_single_model(
            model_name, train_split, val_split, 0, Config.OUTPUT_DIR
        )
        
        # Get test predictions
        test_probs = get_predictions(trainer, tokenizer, test_data)
        all_test_predictions.append(test_probs)
        
        # Free memory
        del trainer, tokenizer
        torch.cuda.empty_cache()
    
    # Ensemble predictions
    print("\n🎯 Combining predictions...")
    
    # Method 1: Majority voting
    mv_preds = majority_voting(all_test_predictions)
    
    # Method 2: Confidence-weighted average
    cw_preds, weights = confidence_weighted(all_test_predictions)
    print(f"   Confidence weights: {[f'{w:.3f}' for w in weights]}")
    
    # Use confidence-weighted as final
    final_preds = cw_preds
    
    # Format predictions
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            'id': item['id'],
            'validity': bool(final_preds[i])
        })
    
    # Save predictions
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✅ Predictions saved to: {Config.PREDICTION_PATH}")
    
    # Create zip
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    print(f"✅ Submission zip created: {Config.SUBMISSION_ZIP_PATH}")
    
    # Statistics
    valid_count = sum(1 for p in predictions if p['validity'])
    print(f"\n📊 Statistics:")
    print(f"   Valid: {valid_count} ({100*valid_count/len(predictions):.1f}%)")
    print(f"   Invalid: {len(predictions) - valid_count}")
    
    # Compare methods
    print("\n📊 Method Agreement:")
    agreement = sum(1 for a, b in zip(mv_preds, cw_preds) if a == b)
    print(f"   Majority vs Confidence-weighted: {100*agreement/len(mv_preds):.1f}% agreement")
    
    print("\n" + "=" * 70)
    print("🎉 Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
