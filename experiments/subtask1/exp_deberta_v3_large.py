"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Syllogistic Reasoning in English
================================================================================
Description: Fine-tune DeBERTa-v3-large for binary classification of syllogism
             validity. Uses stratified split with content effect analysis.
             
Model: microsoft/deberta-v3-large
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
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 
                          'protobuf==3.20.3'])
    
    # Install other packages (most are already on Kaggle)
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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
class Config:
    # Paths - Kaggle environment
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/deberta_subtask1'
    
    # Competition output format
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Model
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    
    # Training - optimized for H100
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 5
    WARMUP_RATIO = 0.1
    
    # Validation
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. DATA LOADING
# ============================================================================
def load_data():
    """Load training and test data."""
    print("📂 Loading data...")
    
    with open(Config.TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"   Train samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    return train_data, test_data

def analyze_data(data):
    """Analyze data distribution by validity and plausibility."""
    stats = {
        (True, True): 0,   # valid, plausible
        (True, False): 0,  # valid, implausible
        (False, True): 0,  # invalid, plausible
        (False, False): 0  # invalid, implausible
    }
    
    for item in data:
        key = (item['validity'], item['plausibility'])
        stats[key] += 1
    
    print("\n📊 Data Distribution (validity, plausibility):")
    for key, count in stats.items():
        print(f"   {key}: {count} ({100*count/len(data):.1f}%)")
    
    return stats

# ============================================================================
# 5. DATASET CLASS
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
        
        # Create input prompt
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
            'id': item['id']
        }
        
        if not self.is_test:
            result['labels'] = torch.tensor(1 if item['validity'] else 0, dtype=torch.long)
        
        return result

# ============================================================================
# 6. STRATIFIED SPLIT
# ============================================================================
def create_stratified_split(train_data, val_ratio=0.1):
    """Create stratified train/val split based on validity AND plausibility."""
    # Create combined stratification key
    strat_keys = [f"{item['validity']}_{item['plausibility']}" for item in train_data]
    
    train_split, val_split = train_test_split(
        train_data,
        test_size=val_ratio,
        stratify=strat_keys,
        random_state=Config.SEED
    )
    
    print(f"\n✅ Train/Val Split:")
    print(f"   Training: {len(train_split)}")
    print(f"   Validation: {len(val_split)}")
    
    return train_split, val_split

# ============================================================================
# 7. METRICS
# ============================================================================
def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def compute_content_effect(predictions, ground_truth):
    """
    Compute Total Content Effect (TCE) metric.
    Lower is better - indicates less bias from content plausibility.
    """
    # Create ground truth map
    gt_map = {item['id']: item for item in ground_truth}
    
    # Group accuracies by (validity, plausibility)
    groups = {
        (True, True): {'correct': 0, 'total': 0},
        (True, False): {'correct': 0, 'total': 0},
        (False, True): {'correct': 0, 'total': 0},
        (False, False): {'correct': 0, 'total': 0}
    }
    
    for pred in predictions:
        gt = gt_map.get(pred['id'])
        if gt:
            key = (gt['validity'], gt['plausibility'])
            groups[key]['total'] += 1
            if pred['validity'] == gt['validity']:
                groups[key]['correct'] += 1
    
    # Calculate accuracies
    accs = {}
    for key, val in groups.items():
        accs[key] = (val['correct'] / val['total'] * 100) if val['total'] > 0 else 0
    
    # Intra-validity: difference within same validity label
    intra_valid = abs(accs[(True, True)] - accs[(True, False)])
    intra_invalid = abs(accs[(False, True)] - accs[(False, False)])
    intra_effect = (intra_valid + intra_invalid) / 2
    
    # Inter-validity: difference across validity labels
    inter_plausible = abs(accs[(True, True)] - accs[(False, True)])
    inter_implausible = abs(accs[(True, False)] - accs[(False, False)])
    inter_effect = (inter_plausible + inter_implausible) / 2
    
    # Total Content Effect
    tce = (intra_effect + inter_effect) / 2
    
    return {
        'tce': tce,
        'intra_effect': intra_effect,
        'inter_effect': inter_effect,
        'group_accuracies': accs
    }

# ============================================================================
# 8. TRAINING
# ============================================================================
def train_model(train_data, val_data, tokenizer):
    """Fine-tune DeBERTa model."""
    print("\n🚀 Starting training...")
    
    # Create datasets
    train_dataset = SyllogismDataset(train_data, tokenizer, Config.MAX_LENGTH)
    val_dataset = SyllogismDataset(val_data, tokenizer, Config.MAX_LENGTH)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Training arguments - optimized for H100 and disk space
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
        save_total_limit=1,  # Only keep 1 checkpoint to save disk space
        save_only_model=True,  # Don't save optimizer state
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=50,
        fp16=True,  # Mixed precision for H100
        dataloader_num_workers=4,
        report_to='none',
        seed=Config.SEED,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    trainer.train()
    
    # Clean up checkpoints to free disk space
    import shutil
    if os.path.exists(Config.OUTPUT_DIR):
        for item in os.listdir(Config.OUTPUT_DIR):
            item_path = os.path.join(Config.OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith('checkpoint'):
                shutil.rmtree(item_path)
                print(f"   Cleaned up: {item}")
    
    print("✅ Training completed!")
    return trainer

# ============================================================================
# 9. INFERENCE
# ============================================================================
def predict(trainer, test_data, tokenizer):
    """Generate predictions for test data."""
    print("\n🔮 Generating predictions...")
    
    test_dataset = SyllogismDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    # Get predictions
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_labels = np.argmax(logits, axis=1)
    
    # Format predictions
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            'id': item['id'],
            'validity': bool(predicted_labels[i])
        })
    
    # Save predictions as predictions.json
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Predictions saved to: {Config.PREDICTION_PATH}")
    
    # Create predictions.zip for submission
    import zipfile
    with zipfile.ZipFile(Config.SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.PREDICTION_PATH, 'predictions.json')
    
    print(f"✅ Submission zip created: {Config.SUBMISSION_ZIP_PATH}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Valid predictions: {sum(1 for p in predictions if p['validity'])}")
    print(f"   Invalid predictions: {sum(1 for p in predictions if not p['validity'])}")
    
    return predictions

# ============================================================================
# 10. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("SemEval-2026 Task 11 - Subtask 1: Syllogistic Reasoning")
    print("Model: DeBERTa-v3-large")
    print("=" * 70)
    
    # Load data
    train_data, test_data = load_data()
    
    # Analyze training data
    analyze_data(train_data)
    
    # Create stratified split
    train_split, val_split = create_stratified_split(train_data, Config.VAL_SPLIT)
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Train model
    trainer = train_model(train_split, val_split, tokenizer)
    
    # Evaluate on validation set with content effect analysis
    print("\n📊 Validation Performance:")
    val_predictions = []
    val_dataset = SyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH, is_test=True)
    val_output = trainer.predict(val_dataset)
    val_labels = np.argmax(val_output.predictions, axis=1)
    
    for i, item in enumerate(val_split):
        val_predictions.append({
            'id': item['id'],
            'validity': bool(val_labels[i])
        })
    
    # Compute content effect
    ce_results = compute_content_effect(val_predictions, val_split)
    overall_acc = accuracy_score(
        [item['validity'] for item in val_split],
        [pred['validity'] for pred in val_predictions]
    ) * 100
    
    print(f"   Overall Accuracy: {overall_acc:.2f}%")
    print(f"   Total Content Effect (TCE): {ce_results['tce']:.2f}")
    print(f"   Combined Score: {overall_acc / (1 + np.log(1 + ce_results['tce'])):.2f}")
    print("\n   Accuracy by (validity, plausibility):")
    for key, acc in ce_results['group_accuracies'].items():
        print(f"      {key}: {acc:.2f}%")
    
    # Generate test predictions
    predictions = predict(trainer, test_data, tokenizer)
    
    print("\n" + "=" * 70)
    print("🎉 Done! Submission file ready.")
    print("=" * 70)
    
    return predictions

if __name__ == '__main__':
    main()
