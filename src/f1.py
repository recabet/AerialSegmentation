import numpy as np

def f1_score(precision:np.array, recall:np.array)->np.array:
    return [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
