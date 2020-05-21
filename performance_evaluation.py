# performance function to eval correlations between true and pred

from sklearn.metrics import make_scorer,accuracy_score,precision_score,recall_score,roc_auc_score, confusion_matrix
import torch
import numpy as np

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")
def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

def performance_function(y_true,y_pred_proba):
    if isinstance(y_pred_proba,torch.Tensor):
        y_pred = np.array(y_pred_proba)
    else:
        y_pred = y_pred_proba
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0

    return [auc(y_true,y_pred_proba),recall(y_true,y_pred),sp(y_true,y_pred),accuracy(y_true,y_pred)]