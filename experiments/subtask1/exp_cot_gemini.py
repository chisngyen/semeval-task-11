"""
================================================================================
SemEval-2026 Task 11 - Subtask 1: Experiment 2 - Chain-of-Thought with Gemini
================================================================================
Description: Zero-shot Chain-of-Thought prompting with Gemini 2.5 Flash.
             Inspired by SymbCoT (ACL 2024) - uses symbolic reasoning steps.
             
Method: CoT prompting with logical analysis before answer
Hardware: Kaggle H100 GPU (uses Gemini API)
================================================================================
"""

# ============================================================================
# 1. INSTALL DEPENDENCIES  
# ============================================================================
import subprocess
import sys

def install_packages():
    packages = [
        'google-generativeai>=0.8.0',
        'tqdm>=4.66.0',
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

install_packages()

# ============================================================================
# 2. IMPORTS
# ============================================================================
import json
import os
import time
import re
from tqdm import tqdm
import google.generativeai as genai

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
class Config:
    # Paths - Kaggle environment
    TRAIN_PATH = '/kaggle/input/semeval-task11/train_data/subtask 1/train_data.json'
    TEST_PATH = '/kaggle/input/semeval-task11/test_data/subtask 1/test_data_subtask_1.json'
    
    # Competition output format
    PREDICTION_PATH = '/kaggle/working/predictions.json'
    SUBMISSION_ZIP_PATH = '/kaggle/working/predictions.zip'
    
    # Gemini API - Add your keys here or use Kaggle secrets
    API_KEYS = [
    # Chis
    'AIzaSyBy8PEjQTIAxHVemYGvEw5qYjDIlHoO3-A',
    'AIzaSyAJw8t0hr5QA9frBvuPAxyn_3D5-BwbXMk',
    'AIzaSyBdIzZ_KBjRFULoZwiqNSIkpiAok-LaLSs',
    'AIzaSyAICetlh9BINq9mY9ohBdNTheUzZDj9Dfc',
    'AIzaSyB8dffnb0pSC1ovLmKTKTNXhu8SH2aMS84',
    'AIzaSyCA7DegHYrYBPCIr_DfMXINihm37DOGlV4',
    'AIzaSyC1n6v35BhnR_1YVAc57hXORXvYkzjXOHs',
    # Phú Hòa
    'AIzaSyBwOUcChNbbBgQ8xRILqyS3Tmlx3OyN-GU',
    'AIzaSyBlSZpvhen8_s7L8ZgB2nzQ0ngLkRHKeC0',
    'AIzaSyCUIetjl_AuL3P_Bah3Qdh4WMDMfbvDHgg',
    # Phú Quý
    'AIzaSyBEdcSUQEoJ714TLC2WB9E2uqNUobENQOg',
    'AIzaSyDiazr0iMuFuvr-ybmRIVtrjuX0r06Ucrs',
    'AIzaSyB0_ODfKIevF2UgjkWfEwKxG3zIRaRY8Gg',
    'AIzaSyD2v1Y3Pem3Txhwmy-MSFegFy9Ul6Us7Ps',
    'AIzaSyCuKKg4HzINkfuaTEO5vJwyG5UezbfSVKs',
    # Trunk
    'AIzaSyBshkYJg6uC12HCyWhhCJbrTRlRngbKIyo',
    'AIzaSyC_OuD6b4AhRXrPd9b5eilBnX4LUUdxydo',
    'AIzaSyAx_TEB9mZa-4reVRc7ydu3bzq2hAK9JFo',
    'AIzaSyCDi1IfbZTUer8Gf6PUvR798_8BVk7g-iQ',
    'AIzaSyDPYIMvlT1PRu3Ok7KH4UFpocqAbhhj2tI',
    'AIzaSyCjX0Sz5fVmY4vECfj5bRI3y7GfWRPlIXA',
    'AIzaSyBC9r1nFqfLeFr7lSI9pBr3TS7qmnmahSI',
    'AIzaSyDcJtxlSuLuh9rA_R5IjfPaf5-oyK8rLMU',
    'AIzaSyAQ2jd3igkiL1XvBxaAbTvkMJxVkje5FzY',
]
    
    MODEL_NAME = 'gemini-2.0-flash'
    
    # Rate limiting
    DELAY_BETWEEN_CALLS = 0.5  # seconds
    MAX_RETRIES = 3

# ============================================================================
# 4. API KEY ROTATION
# ============================================================================
class APIKeyRotator:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_idx = 0
        self.configure_current_key()
    
    def configure_current_key(self):
        if self.api_keys:
            genai.configure(api_key=self.api_keys[self.current_idx])
    
    def rotate(self):
        self.current_idx = (self.current_idx + 1) % len(self.api_keys)
        self.configure_current_key()
        print(f"   Rotated to API key {self.current_idx + 1}/{len(self.api_keys)}")

# ============================================================================
# 5. CHAIN-OF-THOUGHT PROMPT
# ============================================================================
COT_PROMPT = """You are an expert in formal logic and syllogistic reasoning.

A syllogism is a logical argument where a conclusion follows necessarily from two premises based on their logical structure, NOT their real-world plausibility.

**IMPORTANT**: Ignore whether the premises or conclusion are true in the real world. Focus ONLY on whether the conclusion LOGICALLY FOLLOWS from the premises.

**Syllogism to analyze:**
{syllogism}

**Step-by-step analysis:**
1. Identify the two premises and the conclusion
2. Identify the major term, minor term, and middle term  
3. Determine the logical form (mood and figure)
4. Check if the conclusion follows necessarily from the premises
5. Ignore real-world plausibility - focus only on logical structure

**Validity Rules to check:**
- The middle term must be distributed at least once
- No term can be distributed in the conclusion unless distributed in a premise
- At least one premise must be affirmative
- If one premise is negative, the conclusion must be negative
- If both premises are universal, the conclusion cannot be particular (without existential import)

After your analysis, respond with ONLY one of the following:
VALID
INVALID

Your response:"""

# ============================================================================
# 6. INFERENCE FUNCTION
# ============================================================================
def analyze_syllogism(syllogism, model, rotator, retries=0):
    """Analyze a syllogism using CoT prompting."""
    prompt = COT_PROMPT.format(syllogism=syllogism)
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().upper()
        
        # Extract final answer
        if 'VALID' in text.split('\n')[-1]:
            if 'INVALID' in text.split('\n')[-1]:
                return 'INVALID' in text.split('\n')[-1].replace('VALID', '')
            return True
        elif 'INVALID' in text.split('\n')[-1]:
            return False
        else:
            # Fallback: search entire response
            lines = text.split('\n')
            for line in reversed(lines):
                if 'INVALID' in line:
                    return False
                if 'VALID' in line:
                    return True
            return False  # Default to invalid if unclear
            
    except Exception as e:
        if retries < Config.MAX_RETRIES:
            print(f"   Error: {e}, rotating API key...")
            rotator.rotate()
            time.sleep(2)
            return analyze_syllogism(syllogism, model, rotator, retries + 1)
        else:
            print(f"   Max retries reached, defaulting to False")
            return False

# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("SemEval-2026 Task 11 - Subtask 1: CoT with Gemini")
    print("=" * 70)
    
    # Load test data
    print("\n📂 Loading test data...")
    with open(Config.TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"   Loaded {len(test_data)} test samples")
    
    # Initialize API
    if not Config.API_KEYS:
        # Try to get from Kaggle secrets
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            api_key = secrets.get_secret("GEMINI_API_KEY")
            Config.API_KEYS = [api_key]
        except:
            print("❌ No API keys found! Please add your Gemini API keys to Config.API_KEYS")
            return
    
    rotator = APIKeyRotator(Config.API_KEYS)
    model = genai.GenerativeModel(Config.MODEL_NAME)
    
    # Generate predictions
    print("\n🔮 Generating predictions with CoT...")
    predictions = []
    
    for item in tqdm(test_data, desc="Processing"):
        validity = analyze_syllogism(item['syllogism'], model, rotator)
        predictions.append({
            'id': item['id'],
            'validity': validity
        })
        time.sleep(Config.DELAY_BETWEEN_CALLS)
    
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
    print(f"   Invalid: {len(predictions) - valid_count} ({100*(len(predictions)-valid_count)/len(predictions):.1f}%)")
    
    print("\n" + "=" * 70)
    print("🎉 Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
