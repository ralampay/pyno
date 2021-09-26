import os
import numpy as np
from math import sqrt
from sklearn.metrics import confusion_matrix

def performance_metrics(validation_labels, predictions):
    tn, fp, fn, tp = np.array(confusion_matrix(validation_labels, predictions).ravel(), dtype=np.float64)

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    ts  = tp / (tp + fn + fp)
    pt  = (sqrt(tpr * (-tnr + 1)) + tnr - 1) / (tpr + tnr - 1)
    f1  = tp / (tp + 0.5 * (fp + fn))
    acc = (tp + tn) / (tp + tn + fp + fn)

    mcc_denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if mcc_denominator > 0:
        mcc = ((tp * tn) - (fp * fn))  / mcc_denominator
    else:
        mcc = -1

    return {
        'tp':   int(tp),
        'tn':   int(tn), 
        'fp':   int(fp), 
        'fn':   int(fn), 
        'tpr':  tpr, 
        'tnr':  tnr,
        'fpr':  fpr,
        'fnr':  fnr,
        'ppv':  ppv, 
        'npv':  npv, 
        'ts':   ts, 
        'pt':   pt, 
        'acc':  acc, 
        'f1':   f1, 
        'mcc':  mcc
    }
