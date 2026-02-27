# SemEval-2026 Task 11 - Subtask 1 Leaderboard

## Our Experiments

| Exp | Method | Accuracy | Content Effect | Combined Score |
|-----|--------|----------|----------------|----------------|
| 36 | **Ensemble Mean Prob (10x Seeds)** | **100.00%** | **0.00** | **100.00** 🏆 GLOBAL CHAMPION |
| 28b | **Qwen 14B High Rank (r=64)** | **99.48%** | **1.06** | **57.68** 🌍 TOP 3 WORLD |
| 28 | Qwen 14B + Augmentation | 98.43% | 2.13 | 45.99 |
| 33 | Qwen 32B High Rank (r=64) | 97.91% | 3.19 | 40.24 ❌ (Scaling Failed) |
| 33 | Qwen 32B Rank 32 | 97.38% | 3.17 | 40.11 ❌ (Scaling Failed) |
| 31 | Exp28b + TTA (Premise Swap) | 95.29% | 5.21 | 33.72 ❌ (TTA Failed) |
| 26 | Qwen 2.5 14B (LoRA Upgraded) | 97.91% | 2.12 | 45.74 |
| 32 | Exp28b + Structure Prompt | 98.43% | 3.19 | 40.46 ❌ (Prompt Regressed) |
| 27 | Qwen 2.5 7B + Exp21 Augmentation | 98.43% | 3.19 | 40.46 |
| 21 | Valid->Invalid Augmentation (Fixed Exp17) | 95.29% | 3.21 | 39.08 |
| 17 | Counterfactual Augmentation | 92.67% | 3.13 | 38.34 |
| 9 | Exp5+Exp6 (R-Drop + Label Smoothing) | 93.72% | 4.28 | 35.19 |
| 1 | DeBERTa-v3-large (baseline) | 84.82% | 3.32 | 34.42 |
| 25 | **Qwen 2.5 7B (LoRA) + Exp12 Prompt** | 96.34% | 5.32 | 33.88 |
| 12 | Logic-Aware Prompt | 96.34% | 5.32 | 33.88 |
| 14 | Exp9 + Exp12 Combined | 96.86% | 6.36 | 32.33 |
| 4 | Ensemble (DeBERTa + RoBERTa) | 92.67% | 5.36 | 32.51 |
| 5 | Better Hyperparameters | 89.53% | 5.39 | 31.37 |
| 10 | Focal Loss | 91.62% | 6.38 | 30.55 |
| 6 | R-Drop Regularization | 91.62% | 6.43 | 30.49 |
| 18 | Bias Consistency Training (BCT) | 92.67% | 7.47 | 29.55 ❌ |
| 8 | Contrastive + Bias Penalty | 88.48% | 6.45 | 29.41 |
| 20 | Exp17 + Match Exp9 (R-Drop) | 94.24% | 9.57 | 28.06 ❌ |
| 16 | Abstract Symbol Augmentation | 92.67% | 10.57 | 26.87 ❌ |
| 13 | Exp9 Improved (failed) | 87.96% | 9.60 | 26.17 ❌ |
| 19 | Exp17 + Exp12 (Logic Prompt) | 88.48% | 10.66 | 25.60 ❌ |
| 7 | Multi-task (validity + plausibility) | 88.48% | 10.66 | 25.60 |
| 3 | Symbolic FOL Prompt | 79.06% | 8.58 | 24.26 |
| 24 | ALBERT-xxlarge (Exp24) | 49.74% | 50.00 | 10.09 ❌ |
| 2 | Gemini 2.0 Flash (CoT) | 49.74% | 50.00 | 10.09 ❌ |

**🏆 Best Combined Score:** Exp 21 = **39.08** (ACC=95.29%, TCE=3.21)
**🎯 Highest Accuracy:** Exp 14 = **96.86%** (but higher TCE)

## Pending Experiments

| Exp | Method | Status |
|-----|--------|--------|
| 15 | Balanced Sampling + Strong Debiasing | 🔄 Ready |
| 22 | Exp21 + Conflict Reweighting | 🔄 Ready |
| 23 | Exp21 with RoBERTa-large | 🔄 Ready |
| 24 | **ALBERT-xxlarge (Exp24)** | 49.74% | 50.00 | 10.09 ❌ (Failed) |
| 25 | **Qwen 2.5 7B (LoRA 4-bit)** | 🔄 Ready / H100 |
| 26 | **Qwen 2.5 14B (LoRA 4-bit) UPGRADED** | 🔄 Ready / H100 |
| 27 | **Qwen 2.5 7B + Exp21 Augmentation** | 🔄 Ready / H100 |
| 18 | Bias Consistency Training (BCT) | 🔄 Ready |
| 18 | Bias Consistency Training (BCT) | 🔄 Ready |

## CodaBench Leaderboard (Top 10)

*Updated: January 14, 2026*

| Rank | Team | Combined Score | Accuracy | Content Effect |
|------|------|----------------|----------|----------------|
| 1 | rakshith2202 | **58.05** | 99.48% | 1.04 |
| 2 | Habib_TAZ | 58.05 | 99.48% | 1.04 |
| 3 | dutir_shlee | 57.68 | 99.48% | 1.06 |
| 4 | butasrafael | 57.38 | 98.95% | 1.06 |
| 5 | junhaofu | 57.07 | 98.43% | 1.06 |
| 6 | rongchuan | 46.39 | 98.95% | 2.11 |
| 7 | joanitolopo | 46.23 | 98.95% | 2.13 |
| 8 | wahed_e3faa_w_tlata_3saker | 46.23 | 98.95% | 2.13 |
| 9 | Stanford MLab (11b) | 43.26 | 93.19% | 2.17 |
| 10 | zhyuxie | 39.59 | 96.34% | 3.19 |

## Analysis

**Gap to Top 1:**
- Our best (Exp9): Combined Score = 35.19
- Top 1 (rakshith2202): Combined Score = 58.05
- **Gap: 22.86 points**

**Key Insights:**
- Exp9 (R-Drop + Label Smoothing + Cosine LR) achieved best combined score
- Exp12/14 have higher accuracy but higher TCE → lower combined score
- **Problem:** When ACC ↑, TCE also ↑ (model learns content shortcuts)
- Top teams achieve ~99% ACC with TCE ~1.0 - fundamentally different approach

**Strategy:**
- Need to reduce TCE to <2.0 while keeping ACC >95%
- Exp15-18 focus on DEBIASING techniques

