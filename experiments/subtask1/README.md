# Subtask 1 Experiments: Syllogistic Reasoning in English

## Task Description
Binary classification to determine the formal **validity** of syllogisms, independent of content plausibility.

## Experiments

| Exp | File | Method | Difficulty | Status |
|-----|------|--------|------------|--------|
| 1 | `exp_deberta_v3_large.py` | DeBERTa-v3-large baseline | Easy | ✅ 34.42 |
| 2 | `exp_cot_gemini.py` | Gemini 2.0 Flash (CoT) | Easy | ❌ 10.09 |
| 3 | `exp_symbolic_fol.py` | Symbolic FOL Prompt | Medium | ✅ 24.26 |
| 4 | `exp_ensemble.py` | Ensemble (DeBERTa + RoBERTa) | Medium | ✅ 32.51 |
| 5 | `exp5_better_hparams.py` | Better hyperparameters | Easy | ✅ 31.37 |
| 6 | `exp6_rdrop.py` | R-Drop regularization | Medium | ✅ 30.49 |
| 7 | `exp7_multitask.py` | Multi-task (validity + plausibility) | Medium | ✅ 25.60 |
| 8 | `exp8_contrastive_bias.py` | Contrastive + Bias penalty | Hard | 🔄 Ready |
| 9 | `exp9_combined.py` | Exp5 + Exp6 combined | Medium | 🔄 Ready |
| 10 | `exp10_focal_loss.py` | Focal Loss | Easy | 🔄 Ready |
| 11 | `exp11_gradient_reversal.py` | Adversarial debiasing | Hard | 🔄 Ready |
| 12 | `exp12_logic_aware_prompt.py` | Logic-aware structured prompt | Medium | 🔄 Ready |

## Key Metrics
- **ACC**: Overall accuracy (%)
- **TCE**: Total Content Effect (lower = less bias)
- **Combined Score**: `ACC / (1 + ln(1 + TCE))` - Primary ranking metric

## Output
- Predictions: `/kaggle/working/predictions.json`
- Submission: `/kaggle/working/predictions.zip`

## References
- SymbCoT (ACL 2024): Symbolic Chain-of-Thought
- NeuBAROCO (ACL 2024): Syllogism bias evaluation
- LINC (EMNLP 2023): First-order logic prover
