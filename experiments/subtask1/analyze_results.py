
import json
import os
import pandas as pd

RESULTS_DIR = r'd:\semeval_2026_task_11\results'
OUTPUT_FILE = r'd:\semeval_2026_task_11\experiments\subtask1\analysis_report.txt'

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {item['id']: item['validity'] for item in data}

def main():
    output_lines = []
    def log(text=""):
        print(text)
        output_lines.append(text)

    log("Analyzing Exp36 Result Files (Ensembles + Seeds)...")
    
    # Identify all JSON files in results dir
    all_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json') and not f.startswith('__')]
    all_files.sort()
    
    if not all_files:
        log("No result files found.")
        return

    predictions = {}
    for f in all_files:
        path = os.path.join(RESULTS_DIR, f)
        predictions[f] = load_predictions(path)
        log(f"Loaded {f}: {len(predictions[f])} samples")

    sample_ids = sorted(list(predictions[all_files[0]].keys()))
    
    # 1. Agreement Matrix
    log("\n" + "="*50)
    log("PAIRWISE AGREEMENT MATRIX (%)")
    log("="*50)
    
    matrix = pd.DataFrame(index=all_files, columns=all_files)
    for f1 in all_files:
        for f2 in all_files:
            agree = sum(predictions[f1][sid] == predictions[f2][sid] for sid in sample_ids)
            matrix.loc[f1, f2] = round((agree / len(sample_ids)) * 100, 2)
            
    log(matrix.to_string())

    # 2. Detailed Disagreements
    log("\n" + "="*50)
    log("DISAGREEMENT ANALYSIS")
    log("="*50)
    
    disagreement_count = 0
    
    for sid in sample_ids:
        # Get votes for this sample from all files
        votes = {f: predictions[f][sid] for f in all_files}
        unique_votes = set(votes.values())
        
        if len(unique_votes) > 1:
            disagreement_count += 1
            log(f"\n[ Sample ID: {sid} ]")
            log("-" * 20)
            
            # Group files by vote
            valid_files = [f for f, v in votes.items() if v is True]
            invalid_files = [f for f, v in votes.items() if v is False]
            
            log(f"VALID ({len(valid_files)}): {valid_files}")
            log(f"INVALID ({len(invalid_files)}): {invalid_files}")

    if disagreement_count == 0:
        log("\nNo disagreements found! All files are identical.")
    else:
        log(f"\nTotal samples with disagreements: {disagreement_count}")

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
        
    print(f"\nAnalysis saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
