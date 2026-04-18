from typing import List, Tuple

def accuracy(predictions: List[str], gold: List[str]) -> float:
    """Compute accuracy as the fraction of correct predictions."""
    # TODO: implement
    raise NotImplementedError

def precision_recall_f1(predictions: List[str], gold: List[str], positive_label: str = "positive") -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for the positive class.
    
    Returns:
        (precision, recall, f1) as a tuple of floats.
    """
    # TODO: implement without sklearn
    raise NotImplementedError

def confusion_matrix_str(predictions: List[str], gold: List[str], labels: List[str]) -> str:
    """
    Return a pretty-printed confusion matrix string.
    
    Example output:
              Pred Pos  Pred Neg
    True Pos     42        8
    True Neg      5       45
    """
    # TODO: implement
    raise NotImplementedError