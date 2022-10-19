import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
# from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import numpy as np
import math
import time
from sklearn import metrics

import warnings as wrn
wrn.filterwarnings('ignore')


def beam_eval_report(trg_words, topk_prediction_list):
    y_true = np.array(trg_words)
    y_pred = np.array(topk_prediction_list)[:, 0]

    LABELS = np.array(set(list(set(y_true)) + list(set(y_pred))))

    PR = metrics.precision_score(y_true, y_pred, average='weighted')
    RE = metrics.recall_score(y_true, y_pred, average='weighted')
    F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    F05 = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ACC = metrics.accuracy_score(y_true, y_pred)

    print("Evaluation report of beam decoding")
    print(f'''
    Top-1 (Beam Decoding)
        Precision: {PR:.4f}
        Recall: {RE:.4f}
        F1 Score: {F1:.4f}
        F0.5 Score: {F05:.4f}
        Accuracy: {RE * 100:.2f}%
    ''')


def greedy_eval_report(correct_words, predicted_words):
    y_true = np.array(correct_words)
    y_pred = np.array(predicted_words)
    PR = metrics.precision_score(y_true, y_pred, average='weighted')
    RE = metrics.recall_score(y_true, y_pred, average='weighted')
    F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    F05 = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ACC = metrics.accuracy_score(y_true, y_pred)
    print("Evaluation report of greedy decoding")
    print(f'''
    Top-1 (Greedy Decoding)
        Precision: {PR:.4f}
        Recall: {RE:.4f}
        F1 Score: {F1:.4f}
        F0.5 Score: {F05:.4f}
        Accuracy: {RE * 100:.2f}%
    ''')
