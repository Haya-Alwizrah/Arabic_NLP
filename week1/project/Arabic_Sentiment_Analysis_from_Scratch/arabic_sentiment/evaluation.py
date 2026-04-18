from typing import List, Tuple

def accuracy(predictions: List[str], gold: List[str]) -> float:
    """Compute accuracy as the fraction of correct predictions."""
    c = 0
    total = len(gold)

    for p,g in zip(predictions, gold):
        if p ==g:
            c += 1
    score = c / total
    return score


def precision_recall_f1(predictions: List[str], gold: List[str], positive_label: str = "positive") -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for the positive class.
    
    Returns:
        (precision, recall, f1) as a tuple of floats.
    """
    # precision = tp / (tp+fp) --> retrived relevent/ all retrived
    # recall = tp / (tp+fn) -->  retrived relevent/ all relevent
    # f1 = 2* (precision*recall)/(precision+recall) 
    tp = 0 # retrived relevent
    fp = 0 # retrived irelevent
    fn = 0 # not retrived relevent

    for p, g in zip(predictions, gold):
        if p == positive_label and g == positive_label:
            tp += 1
        elif p == positive_label and g != positive_label:
            fp += 1
        elif p != positive_label and g == positive_label:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1




def confusion_matrix_str(predictions: List[str], gold: List[str], labels: List[str]) -> str:
    """
    Return a pretty-printed confusion matrix string.
    
    Example output:
              Pred Pos  Pred Neg
    True Pos     42        8
    True Neg      5       45
    """
    m = {}
    for actual in labels:
        m[actual] = {}
        for pred in labels:
            m[actual][pred] = 0

    for p, g in zip(predictions, gold):
        if p in labels and g in labels:
            m[g][p] += 1

    header = f"{' ':12} | {'Pred ' + labels[0]:10} | {'Pred ' + labels[1]:10}"
    line = "-" * len(header)
    row1 = f"{'True ' + labels[0]:12} | {m[labels[0]][labels[0]]:10} | {m[labels[0]][labels[1]]:10}"
    row2 = f"{'True ' + labels[1]:12} | {m[labels[1]][labels[0]]:10} | {m[labels[1]][labels[1]]:10}"
    
    return f"{header}\n{line}\n{row1}\n{row2}"