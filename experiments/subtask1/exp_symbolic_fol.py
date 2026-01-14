"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 3 - Symbolic Logic Translation
================================================================================
Description: Translate syllogisms to formal logic notation, then evaluate
             validity using logical rules. Inspired by LINC and LOGIC-LM.
             
Method: Fine-tune model to output First-Order Logic representation,
        then verify with symbolic rules.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
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
    # Paths - Kaggle environment
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    OUTPUT_DIR = '/kaggle/working/symbolic_model'
    
    # Competition output format
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Model - Using logic-specialized prompt format
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 384  # Longer for FOL representation
    
    # Training - H100 optimized
    BATCH_SIZE = 24
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 8e-6
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 6
    WARMUP_RATIO = 0.1
    
    # Validation
    VAL_SPLIT = 0.1
    SEED = 42

set_seed(Config.SEED)

# ============================================================================
# 4. SYMBOLIC PROMPT TEMPLATE
# ============================================================================
def create_symbolic_prompt(syllogism):
    """
    Create a prompt that encourages the model to reason symbolically.
    Inspired by SymbCoT and LINC approaches.
    """
    prompt = f"""[LOGIC ANALYSIS]
Analyze the following syllogism for formal logical validity.

[SYLLOGISM]
{syllogism}

[INSTRUCTIONS]
1. Identify premises (P1, P2) and conclusion (C)
2. Extract logical quantifiers: All(∀), Some(∃), No(¬∃), Not all(¬∀)
3. Identify terms: Subject(S), Predicate(P), Middle(M)
4. Check syllogistic figure and mood
5. Apply validity rules

[VALIDITY CHECK]
Focus on LOGICAL STRUCTURE only, ignore real-world truth.
A syllogism is VALID if the conclusion necessarily follows from premises.
A syllogism is INVALID if the conclusion does NOT necessarily follow.

[DETERMINATION]"""
    
    return prompt

# ============================================================================
# 5. DATASET CLASS
# ============================================================================
class SymbolicSyllogismDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create symbolic prompt
        text = create_symbolic_prompt(item['syllogism'])
        
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
# 6. FOCAL LOSS FOR HANDLING CLASS IMBALANCE
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss to handle potential class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# 7. CUSTOM MODEL WITH FOCAL LOSS
# ============================================================================
class SymbolicClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ============================================================================
# 8. TRAINING AND INFERENCE
# ============================================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def train_and_predict():
    print("=" * 70)
    print("SemEval-2026 Task 11 - Subtask 1: Symbolic Logic Approach")
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
    print(f"\n📥 Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Datasets
    train_dataset = SymbolicSyllogismDataset(train_split, tokenizer, Config.MAX_LENGTH)
    val_dataset = SymbolicSyllogismDataset(val_split, tokenizer, Config.MAX_LENGTH)
    test_dataset = SymbolicSyllogismDataset(test_data, tokenizer, Config.MAX_LENGTH, is_test=True)
    
    # Model with Focal Loss
    print("\n🚀 Starting training with Focal Loss...")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Training arguments - optimized for disk space
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
        save_total_limit=1,  # Only keep 1 checkpoint
        save_only_model=True,  # Don't save optimizer state
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=50,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
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
    
    # Inference
    print("\n🔮 Generating predictions...")
    test_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(test_output.predictions, axis=1)
    
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            'id': item['id'],
            'validity': bool(predicted_labels[i])
        })
    
    # Save
    with open(Config.PREDICTION_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"✅ Predictions saved to: {Config.PREDICTION_PATH}")
    
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
    
    print("\n" + "=" * 70)
    print("🎉 Done!")
    print("=" * 70)

if __name__ == '__main__':
    train_and_predict()
