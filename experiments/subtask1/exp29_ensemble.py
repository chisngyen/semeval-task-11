"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 29 - Ensemble (Majority Voting)
================================================================================
Description: Combines predictions from the top 3 models (Exp26, Exp27, Exp28)
             using Majority Voting.
             
             Rationale:
             - Exp26 (Qwen 14B): Low TCE (2.12), High Acc.
             - Exp27 (Qwen 7B + Aug): SOTA Augmentation, High Acc.
             - Exp28 (Qwen 14B + Aug): The "Ultimate" Model.
             
             Ensembling uncorrelated errors typically boosts performance.
             Since we only have hard labels (True/False) from previous runs,
             we use Majority Voting (Hard Ensemble).
             
Usage: 
    1. Rename your prediction files to:
       - pred_exp26.json
       - pred_exp27.json
       - pred_exp28.json
    2. Place them in the same directory (or update paths below).
    3. Run this script.
================================================================================
"""

import json
import os
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
FILES = [
    '/kaggle/working/pred_exp26.json',
    '/kaggle/working/pred_exp27.json',
    '/kaggle/working/pred_exp28.json',
    '/kaggle/working/pred_exp28b.json',
    '/kaggle/working/pred_exp32.json',
    '/kaggle/working/pred_exp33.json',
    '/kaggle/working/pred_exp35.json'
]
OUTPUT_PATH = '/kaggle/working/predictions.json'
ZIP_PATH = '/kaggle/working/predictions.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("Exp 29: Ensemble (Majority Voting)")
    print("=" * 60)
    
    # Load all predictions
    all_preds = []
    for fp in FILES:
        if not os.path.exists(fp):
            print(f"⚠️ Warning: File not found: {fp}")
            continue
            
        print(f"Loading: {fp}")
        with open(fp, 'r') as f:
            data = json.load(f)
            # data is list of {'id': ..., 'validity': ...}
            # Convert to dict for easy lookup
            pred_map = {item['id']: item['validity'] for item in data}
            all_preds.append(pred_map)
            
    if not all_preds:
        print("❌ No prediction files found! Exiting.")
        return

    print(f"\nEnsembling {len(all_preds)} models...")
    
    # Get all IDs (assuming all files have same IDs, use first file as reference)
    example_file = all_preds[0]
    ids = sorted(list(example_file.keys()))
    
    final_preds = []
    
    for _id in ids:
        votes = []
        for model_preds in all_preds:
            if _id in model_preds:
                votes.append(model_preds[_id])
        
        # Majority Vote
        if votes:
            vote_count = Counter(votes)
            # most_common(1) returns [(value, count)]
            # If tie (e.g. 1 True, 1 False), defaulting to True (or first)
            # But with 3 models, ties are impossible unless missing data
            final_validity = vote_count.most_common(1)[0][0]
            final_preds.append({'id': _id, 'validity': final_validity})
        else:
            # Fallback if id missing (should not happen)
            final_preds.append({'id': _id, 'validity': False})
            
    # Save
    print(f"Saving {len(final_preds)} predictions to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_preds, f, indent=2)
        
    import zipfile
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_PATH, 'predictions.json')
        
    print(f"✅ Saved Submission: {ZIP_PATH}")

if __name__ == '__main__':
    main()
