"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 35 - Self-Consistency CoT
================================================================================
Description: Majority Voting over multiple Chain-of-Thought (CoT) reasoning paths.
             
             Method:
             1. Use **Qwen 72B Instruct** (The Teacher).
             2. For each test sample, generate 5 different responses (paths).
                - Temperature = 0.7 (to ensure diversity in reasoning).
             3. Parse each path -> Valid/Invalid.
             4. Final Answer = Majority Vote (e.g., 3 Valid vs 2 Invalid -> Valid).
             
             Rationale:
             - User concern: "Zero-shot 14B < Fine-tuned 14B".
             - Solution: Use 72B (much smarter) for Zero-shot Reasoning.
             - H100 can handle 72B in 4-bit (~48GB VRAM).
             
Model: Qwen/Qwen2.5-72B-Instruct (Int4)
================================================================================
"""

# ============================================================================
# 0. INSTALL DEPENDENCIES
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

import json
import torch
import re
from tqdm import tqdm
from collections import Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    MODEL_NAME = 'Qwen/Qwen2.5-72B-Instruct'
    OUTPUT_FILE = '/kaggle/working/predictions.json'
    ZIP_FILE = '/kaggle/working/predictions.zip'
    
    # Self-Consistency Config
    NUM_PATHS = 5        # Number of reasoning paths per question
    MAX_NEW_TOKENS = 384 # More tokens for detailed reasoning
    TEMPERATURE = 0.7    # High temp for diversity (crucial for Self-Consistency)
    TOP_P = 0.95

# ============================================================================
# LOGIC PROMPT WITH COT (Same as Exp30)
# ============================================================================
def parse_syllogism(text):
    premise1_match = re.search(r'Premise 1[:\s]*(.+?)(?=Premise 2|$)', text, re.IGNORECASE | re.DOTALL)
    premise2_match = re.search(r'Premise 2[:\s]*(.+?)(?=Conclusion|$)', text, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(r'Conclusion[:\s]*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    if premise1_match and premise2_match and conclusion_match:
        return {'premise1': premise1_match.group(1).strip(), 'premise2': premise2_match.group(1).strip(), 'conclusion': conclusion_match.group(1).strip()}
    return {'premise1': text, 'premise2': '', 'conclusion': ''}

def create_cot_prompt(syllogism_text):
    parsed = parse_syllogism(syllogism_text)
    return f"""<|im_start|>system
You are a master logician. Your task is to analyze the following syllogism.
First, perform a step-by-step logical analysis.
Then, give your final verdict: "VALID" or "INVALID".<|im_end|>
<|im_start|>user
[PREMISE 1]: {parsed['premise1']}
[PREMISE 2]: {parsed['premise2']}
[CONCLUSION]: {parsed['conclusion']}

Think step by step.<|im_end|>
<|im_start|>assistant
Logical Analysis:"""

# ============================================================================
# PARSING HELPER
# ============================================================================
def parse_response(response):
    response_lower = response.lower()
    # Priority 1: Check the very end
    last_words = response_lower.split()[-20:]
    if "invalid" in last_words: return False
    if "valid" in last_words: return True
    
    # Priority 2: Clear statement phrases
    if "conclusion is invalid" in response_lower: return False
    if "conclusion is valid" in response_lower: return True
    
    # Fallback
    return False # Conservative default

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print(f"Exp 35: Self-Consistency CoT (k={Config.NUM_PATHS})")
    print("="*60)
    
    # 1. Load Data
    with open(Config.TEST_PATH, 'r') as f:
        test_data = json.load(f)
        
    # 2. Load Model
    print("Loading Qwen 14B Instruct...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    results = []
    print(f"\n🧠 Generating {Config.NUM_PATHS} paths for each of {len(test_data)} samples...")
    
    for item in tqdm(test_data):
        prompt = create_cot_prompt(item['syllogism'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Expand inputs for batch generation of multiple paths
        # Note: Doing this in a loop to avoid OOM if batch*paths is too big
        # Or we can use beam_sample / direct sampling
        
        paths_validity = []
        
        # Generate N paths
        # We can do this in one go with num_return_sequences if VRAM permits,
        # but safely looping is better for 14B on limited VRAM
        for _ in range(Config.NUM_PATHS):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    temperature=Config.TEMPERATURE,
                    do_sample=True, # MUST be True for Self-Consistency
                    top_p=Config.TOP_P,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Parse only the assistant response
            assistant_response = response.split("Logical Analysis:")[-1]
            vote = parse_response(assistant_response)
            paths_validity.append(vote)
            
        # MAJORITY VOTE
        vote_counts = Counter(paths_validity)
        final_verdict = vote_counts.most_common(1)[0][0]
        
        # Optional: Print disagreement cases (debugging)
        # if len(vote_counts) > 1:
        #     print(f"Disagreement: {paths_validity} -> {final_verdict}")
            
        results.append({'id': item['id'], 'validity': final_verdict})
        
    # 3. Save
    with open(Config.OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    import zipfile
    with zipfile.ZipFile(Config.ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.OUTPUT_FILE, 'predictions.json')
        
    print(f"\n✅ Saved Self-Consistency Results: {Config.ZIP_FILE}")

if __name__ == '__main__':
    main()
