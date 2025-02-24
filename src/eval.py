def compute_label_based_accuracy(filepath):
    """
    Reads a text file containing (index, query_id, candidate_id, correct_label, prediction)
    and computes the overall accuracy as well as accuracy by each label (0 or 1).
    
    :param filepath: Path to the prediction results file.
    :return: A dictionary containing accuracy for label 0, label 1, and overall accuracy.
    """
    # Track per-label counts
    total_0 = 0
    correct_0 = 0
    total_1 = 0
    correct_1 = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip the header line
        header = next(f, None)

        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Expecting columns split by tab:
            # index, query_id, candidate_id, correct_label, prediction
            parts = line.split('\t')
            if len(parts) < 5:
                continue  # skip malformed lines

            # Extract the correct_label and prediction as integers
            correct_label = int(parts[3])
            prediction = int(parts[4])

            # Update counts for label 0
            if correct_label == 0:
                total_0 += 1
                if prediction == 0:
                    correct_0 += 1

            # Update counts for label 1
            elif correct_label == 1:
                total_1 += 1
                if prediction == 1:
                    correct_1 += 1

    # Calculate accuracies, guarding against divide-by-zero
    accuracy_0 = correct_0 / total_0 if total_0 else 0.0
    accuracy_1 = correct_1 / total_1 if total_1 else 0.0

    # Overall accuracy
    overall_correct = correct_0 + correct_1
    overall_total = total_0 + total_1
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0

    return {
        "accuracy_label_0": accuracy_0,
        "accuracy_label_1": accuracy_1,
        "overall_accuracy": overall_accuracy,
    }


# Example usage:
if __name__ == "__main__":
    test_file_path = "/home/s2320037/SCIDOCA/outputs/distilbert-base-uncased_batch_16_epoch_30_LR_5e-5_task1/predict_results.txt"
    results = compute_label_based_accuracy(test_file_path)
    print("Accuracy for label 0:", results["accuracy_label_0"])
    print("Accuracy for label 1:", results["accuracy_label_1"])
    print("Overall accuracy:", results["overall_accuracy"])
