import numpy as np


def select_cases(y_true, y_pred):
    """
    Select one TP, FP, FN index.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cases = {}

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    cases["tp"] = tp[0] if len(tp) > 0 else None
    cases["fp"] = fp[0] if len(fp) > 0 else None
    cases["fn"] = fn[0] if len(fn) > 0 else None

    return cases
