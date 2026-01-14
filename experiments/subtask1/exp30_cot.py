"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 30 - Chain-of-Thought (CoT)
================================================================================
Description: Zero-Shot Chain-of-Thought Inference.
             
             Unlike Exp28 (Classification Head), this uses the Generative Head (CausalLM)
             of the Base Qwen 14B Instruct model.
             
             Prompt Strategy:
             "Analyze the logic step-by-step. Is the conclusion Valid?"
             
             Hypothesis:
             Allowing the model to "think" (generate tokens) before answering
             activates its reasoning circuits, potentially correcting edge cases
             where the Classification Head (Exp28) fails.
             
Model: Qwen/Qwen2.5-14B-Instruct (Base, no LoRA adapters aimed here yet)
       (Using Base is better for CoT reasoning than a Classification-tuned adapter)
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
    MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
    OUTPUT_FILE = '/kaggle/working/predictions.json'
    ZIP_FILE = '/kaggle/working/predictions.zip'
    
    # Generation Config
    MAX_NEW_TOKENS = 256 # Allow space for reasoning
    TEMPERATURE = 0.1    # Low temp for deterministic logic

# ============================================================================
# LOGIC PROMPT WITH COT
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
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("Exp 30: Zero-Shot Chain-of-Thought (Qwen 14B)")
    print("="*60)
    
    # 1. Load Data
    with open(Config.TEST_PATH, 'r') as f:
        test_data = json.load(f)
        
    # 2. Load Model (CausalLM)
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
    print(f"Generating CoT for {len(test_data)} samples...")
    
    # 3. Inference Loop
    for item in tqdm(test_data):
        prompt = create_cot_prompt(item['syllogism'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the assistant's part (after logic prompt)
        # Simple parsing strategy: Look for the last occurrence of Valid/Invalid
        response_lower = response.lower()
        
        # Heuristic parsing
        is_valid = None
        if "invalid" in response_lower.split()[-20:]: # Check end of text
             is_valid = False
        elif "valid" in response_lower.split()[-20:]:
             is_valid = True
        else:
            # Fallback scan
            if "therefore, the conclusion is invalid" in response_lower: is_valid = False
            elif "therefore, the conclusion is valid" in response_lower: is_valid = True
            else:
                # Default to False if unsure (Conservative)
                is_valid = False
        
        results.append({'id': item['id'], 'validity': is_valid})
        
    # 4. Save
    with open(Config.OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    import zipfile
    with zipfile.ZipFile(Config.ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(Config.OUTPUT_FILE, 'predictions.json')
        
    print(f"✅ Saved CoT Predictions: {Config.ZIP_FILE}")

if __name__ == '__main__':
    main()
