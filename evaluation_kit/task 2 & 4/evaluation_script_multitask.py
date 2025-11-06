import json
import math
from typing import List, Dict, Any, Tuple, Optional

# --- 1. MOCK DATA FOR DEMONSTRATION ---

# The syllogism for this task can include multiple premises. 
# However, only a subset of premises are truly relevant (i.e., expressed as an index in "premises" indicating their relative position in the syllogism).
# Moreoeve, the ontion of relevance is about logical validity - i.e., the set of valid premises contain the set of sentences that are necessary and sufficient to entail the conclusion. 
# This means that only "valid" syllogisms have relevant premises.
GROUND_TRUTH_DATA_JSON = """
[
    {"id": "id1", "validity": false, "plausibility": true, "premises": [0, 2]},
    {"id": "id2", "validity": true, "plausibility": true, "premises": [1, 3]},
    {"id": "id3", "validity": false, "plausibility": false, "premises": []},
    {"id": "id4", "validity": true, "plausibility": false, "premises": []},
    {"id": "id5", "validity": true, "plausibility": true, "premises": [1, 2]},
    {"id": "id6", "validity": false, "plausibility": true, "premises": [0, 3]},
    {"id": "id7", "validity": false, "plausibility": false, "premises": []},
    {"id": "id8", "validity": true, "plausibility": false, "premises": []},
    {"id": "id9", "validity": true, "plausibility": true, "premises": [2, 3]},
    {"id": "id10", "validity": false, "plausibility": true, "premises": [0, 3]}
]
"""

# Mock Predictions for Model A (High Bias, High Premise F1)
MOCK_PREDICTIONS_MODEL_A_JSON = """
[
    {"id": "id1", "validity": true,  "premises": [0, 2, 3]}, 
    {"id": "id2", "validity": true,  "premises": [1, 3]}, 
    {"id": "id3", "validity": false, "premises": []}, 
    {"id": "id4", "validity": false, "premises": []}, 
    {"id": "id5", "validity": true,  "premises": [1, 2]}, 
    {"id": "id6", "validity": true,  "premises": [0, 3]}, 
    {"id": "id7", "validity": false, "premises": []},
    {"id": "id8", "validity": true,  "premises": [0, 1]}, 
    {"id": "id9", "validity": true,  "premises": [2, 3]}, 
    {"id": "id10", "validity": true, "premises": [0, 3, 2]} 
]
"""

# Mock Predictions for Model B (Lower Bias, Lower Premise F1)
MOCK_PREDICTIONS_MODEL_B_JSON = """
[
    {"id": "id1", "validity": false, "premises": []}, 
    {"id": "id2", "validity": true,  "premises": [1, 3]}, 
    {"id": "id3", "validity": false, "premises": []}, 
    {"id": "id4", "validity": true,  "premises": [2, 3]}, 
    {"id": "id5", "validity": false, "premises": []}, 
    {"id": "id6", "validity": false, "premises": []}, 
    {"id": "id7", "validity": true,  "premises": [1, 2]}, 
    {"id": "id8", "validity": false, "premises": []}, 
    {"id": "id9", "validity": true,  "premises": [2, 3]}, 
    {"id": "id10", "validity": false, "premises": []} 
]
"""

# --- 2. CORE FUNCTIONS ---

def calculate_f1_premises(gt_map: Dict[str, Any], predictions: List[Dict[str, Any]]) -> float:
    """
    Calculates the F1-Score for premise retrieval across all data points.
    Treats each predicted item as a multi-label classification problem.
    """
    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0

    for pred_item in predictions:
        item_id = pred_item['id']
        if item_id in gt_map and 'premises' in gt_map[item_id] and 'premises' in pred_item:
            
            # Ensure lists are sets for efficient comparison
            true_positives = set(gt_map[item_id]['premises'])
            predicted_positives = set(pred_item['premises'])

            # True Positives: Correctly selected relevant premises
            TP = len(true_positives.intersection(predicted_positives))
            # False Positives: Irrelevant premises incorrectly selected
            FP = len(predicted_positives.difference(true_positives))
            # False Negatives: Relevant premises incorrectly missed
            FN = len(true_positives.difference(predicted_positives))

            # Calculate Precision and Recall for this single item
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            # Accumulate item-level scores (Macro-averaging)
            total_precision += precision
            total_recall += recall
            valid_count += 1
    
    if valid_count == 0:
        return 0.0

    # Average Precision and Recall (Macro-Average for component scores)
    macro_precision = total_precision / valid_count
    macro_recall = total_recall / valid_count
    
    # Calculate Macro F1-Score
    f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
    
    # Scale to 100%
    return f1_score * 100

# --- 2. CORE FUNCTIONS FOR ACCURACY AND BIAS CALCULATION ---

def calculate_accuracy(
    ground_truth_list: List[Dict[str, Any]],
    predictions_list: List[Dict[str, Any]],
    metric_name: str,
    prediction_key: str,
    plausibility_filter: Optional[bool] = None 
) -> Tuple[float, int, int]:

    gt_map = {item['id']: item for item in ground_truth_list}
    correct_predictions = 0
    total_predictions = 0
    for pred_item in predictions_list:
        item_id = pred_item['id']
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            gt_plausibility = gt_item.get('plausibility')
            if plausibility_filter is not None and gt_plausibility != plausibility_filter:
                continue 
            if metric_name in gt_item and prediction_key in pred_item:
                true_label = gt_item[metric_name]
                predicted_label = pred_item[prediction_key]
                if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                    total_predictions += 1
                    if true_label == predicted_label:
                        correct_predictions += 1
    if total_predictions == 0:
        return 0.0, 0, 0
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions

def calculate_subgroup_accuracy(
    gt_map: Dict[str, Any],
    predictions_list: List[Dict[str, Any]],
    gt_validity: bool,
    gt_plausibility: bool
) -> Tuple[float, int, int]:

    correct_predictions = 0
    total_predictions = 0
    for pred_item in predictions_list:
        item_id = pred_item['id']
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            if gt_item.get('validity') == gt_validity and gt_item.get('plausibility') == gt_plausibility:
                if 'validity' in gt_item and 'validity' in pred_item:
                    true_label = gt_item['validity']
                    predicted_label = pred_item['validity']
                    if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                        total_predictions += 1
                        if true_label == predicted_label:
                            correct_predictions += 1
    if total_predictions == 0:
        return 0.0, 0, 0
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions

def calculate_content_effect_bias(accuracies: Dict[str, float]) -> Dict[str, float]:

    acc_plausible_valid = accuracies.get('acc_plausible_valid', 0.0)
    acc_implausible_valid = accuracies.get('acc_implausible_valid', 0.0)
    acc_plausible_invalid = accuracies.get('acc_plausible_invalid', 0.0)
    acc_implausible_invalid = accuracies.get('acc_implausible_invalid', 0.0)
    intra_valid_diff = abs(acc_plausible_valid - acc_implausible_valid)
    intra_invalid_diff = abs(acc_plausible_invalid - acc_implausible_invalid)
    content_effect_intra_validity_label = (intra_valid_diff + intra_invalid_diff) / 2.0
    inter_plausible_diff = abs(acc_plausible_valid - acc_plausible_invalid)
    inter_implausible_diff = abs(acc_plausible_valid - acc_implausible_invalid)
    content_effect_inter_validity_label = (inter_plausible_diff + inter_implausible_diff) / 2.0
    tot_content_effect = (content_effect_intra_validity_label + content_effect_inter_validity_label) / 2.0
    return {
        'content_effect_intra_validity_label': content_effect_intra_validity_label,
        'content_effect_inter_validity_label': content_effect_inter_validity_label, # FIXED TYPO HERE
        'tot_content_effect': tot_content_effect
    }

def calculate_smoother_combined_metric(overall_metric: float, total_content_effect: float) -> float:
    """
    Computes the smoother combined score using the natural logarithm.
    Formula: overall_metric / (1 + ln(1 + content_effect))
    """
    if total_content_effect < 0:
        return 0.0

    log_penalty = math.log(1 + total_content_effect)
    smoother_score = overall_metric / (1 + log_penalty)
    return smoother_score


def analyze_model_performance(model_name: str, ground_truth: List[Dict[str, Any]], predictions: List[Dict[str, Any]], gt_map: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full analysis pipeline for a single model's prediction set and returns key metrics."""
    
    print("\n" + "#" * 70)
    print(f"--- RESULTS FOR {model_name} (Accuracy vs. F1_Premises vs. Bias) ---")
    print("#" * 70)

    # --- 1. Premise Retrieval F1-Score ---
    f1_premises = calculate_f1_premises(gt_map, predictions)
    print(f"[Metric: F1_Premises] Macro-Averaged F1: {f1_premises:.2f}%")
    print("-" * 70)

    # --- 2. Validity Accuracy (Reasoning Task) ---
    common_args = {
        'ground_truth_list': ground_truth,
        'predictions_list': predictions,
        'metric_name': 'validity',
        'prediction_key': 'validity'
    }

    overall_acc, overall_correct, overall_total = calculate_accuracy(
        **common_args,
        plausibility_filter=None
    )
    print(f"[Metric: Validity Accuracy] Overall Accuracy: {overall_acc:.2f}% ({overall_correct} / {overall_total})")

    # Grouped Accuracy (for bias calculation)
    acc_plausible, _, total_plausible = calculate_accuracy(**common_args, plausibility_filter=True)
    acc_implausible, _, total_implausible = calculate_accuracy(**common_args, plausibility_filter=False)
    
    print(f"  Plausible Group (N={total_plausible}): {acc_plausible:.2f}%")
    print(f"  Implausible Group (N={total_implausible}): {acc_implausible:.2f}%")

    # --- 3. Content Effect Bias Calculation ---
    print("\n" + "=" * 70)
    print("--- Content Effect Bias Metrics (Based on Validity Accuracy) ---")
    
    # Calculate the four required conditional accuracies
    acc_plausible_valid, _, total_pv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=True)
    acc_implausible_valid, _, total_iv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=False)
    acc_plausible_invalid, _, total_pi = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=True)
    acc_implausible_invalid, _, total_ii = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=False)
    
    conditional_accuracies = {
        'acc_plausible_valid': acc_plausible_valid, 'acc_implausible_valid': acc_implausible_valid,
        'acc_plausible_invalid': acc_plausible_invalid, 'acc_implausible_invalid': acc_implausible_invalid
    }

    bias_metrics = calculate_content_effect_bias(conditional_accuracies)
    tot_content_effect = bias_metrics['tot_content_effect']

    print(f"  Total Content Effect (acc_tot): {tot_content_effect:.2f}%")
    
    # --- 4. Combined Metric Calculation (The Ranking Metric) ---
    
    # Numerator: Average of Validity Accuracy and Premise F1 (Both on 0-100 scale)
    overall_performance_metric = (overall_acc + f1_premises) / 2.0
    
    smoother_score = calculate_smoother_combined_metric(overall_performance_metric, tot_content_effect)
    
    print("\n" + "=" * 70)
    print("--- FINAL RANKING SCORE ---")
    print(f"Overall Performance Metric (Avg(Acc, F1)): {overall_performance_metric:.2f}%")
    print(f"Log-Smoothed Combined Score: {smoother_score:.2f}")
    print("=" * 70)

    return {
        'name': model_name,
        'overall_acc': overall_acc,
        'f1_premises': f1_premises,
        'overall_performance_metric': overall_performance_metric,
        'tot_content_effect': tot_content_effect,
        'smoother_score': smoother_score
    }


# --- 3. EXECUTION ---

if __name__ == "__main__":
    try:
        # Load data from the mock JSON strings
        ground_truth = json.loads(GROUND_TRUTH_DATA_JSON)
        gt_map = {item['id']: item for item in ground_truth}

        # Define the models to analyze
        models_to_analyze = {
            "MODEL A (High Bias, High F1)": json.loads(MOCK_PREDICTIONS_MODEL_A_JSON),
            "MODEL B (Low Bias, Lower F1)": json.loads(MOCK_PREDICTIONS_MODEL_B_JSON),
        }
        
        all_model_results = []

        # Step 1: Analyze models individually and store results
        for model_name, predictions in models_to_analyze.items():
            results = analyze_model_performance(model_name, ground_truth, predictions, gt_map)
            all_model_results.append(results)

        # Step 2: Print Comparison Table
        print("\n\n" + "#" * 80)
        print("--- FINAL RANKING SUMMARY (Based on Log-Smoothed Score) ---")
        print("#" * 80)

        # Sort by Smoother Score (descending for ranking)
        sorted_results = sorted(all_model_results, key=lambda x: x['smoother_score'], reverse=True)

        print("-" * 80)
        print(f"| {'Rank':<4} | {'Model Name':<25} | {'Acc Avg(V,F1)':>13} | {'Total Bias':>10} | {'Smooth Score':>12} |")
        print("-" * 80)
        
        for rank, res in enumerate(sorted_results, 1):
            print(f"| {rank:<4} | {res['name']:<25} | {res['overall_performance_metric']:12.2f}% | {res['tot_content_effect']:9.2f}% | {res['smoother_score']:12.2f} |")
        
        print("-" * 80)
        
        # A more detailed view showing the components
        print("\nDetailed Component Performance:")
        print("-" * 80)
        print(f"| {'Model Name':<25} | {'Validity Acc':>11} | {'F1 Premises':>12} | {'Total Bias':>10} |")
        print("-" * 80)
        
        for res in all_model_results:
            print(f"| {res['name']:<25} | {res['overall_acc']:11.2f}% | {res['f1_premises']:12.2f}% | {res['tot_content_effect']:10.2f}% |")

        print("-" * 80)


    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")