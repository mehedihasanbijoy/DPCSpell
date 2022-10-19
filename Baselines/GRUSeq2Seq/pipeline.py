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
from utils import translate_sentence
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import numpy as np
import math
import time

import warnings as wrn
wrn.filterwarnings('ignore')


def train(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(tqdm(iterator)):
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # print(f"output: {output.shape}, target: {trg.shape} \n\n{trg}")

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(iterator)):
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test_accuracy(test_data, SRC, TRG, model, DEVICE):
    df = pd.read_csv('./Dataset/allDictWords_df.csv')
    # df = pd.read_csv('./Dataset/df_all_words.csv')
    all_words = sorted(df.iloc[:, 0].values)

    erroneous_words, predicted_words, correct_words, flags = [], [], [], []
    modified_flags = []
    for idx, data in enumerate(tqdm(test_data)):
        src = data.src
        trg = data.trg
        translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)

        src = ''.join(src)
        trg = ''.join(trg)
        pred = ''.join(translation[:-1])

        erroneous_words.append(src)
        predicted_words.append(pred)
        correct_words.append(trg)
        if trg == pred:
            flags.append(1)
        else:
            flags.append(0)

        if pred in all_words:
            modified_flags.append(1)
        else:
            modified_flags.append(0)

    modified_acc = np.sum(modified_flags) / len(modified_flags)

    evaluation_df = pd.DataFrame({
        'Error': erroneous_words,
        'Predicton': predicted_words,
        'Target': correct_words,
        'Correction': flags
    })
    # evaluation_df.to_csv('/content/drive/MyDrive/Bangla Spell & Grammar Checker/Codes/GEDC/Seq2Seq/preds_greedy.csv', index=False)

    corrected_instances = evaluation_df['Correction'].values.sum()
    total_instances = len(evaluation_df)
    accuracy = corrected_instances / total_instances
    print(f"\nCorrection/Total: {corrected_instances} / {total_instances}")
    # print(f"Accuracy: {accuracy*100:.2f}%")

    y_true = np.array(correct_words)
    y_pred = np.array(predicted_words)
    PR = metrics.precision_score(y_true, y_pred, average='weighted')
    RE = metrics.recall_score(y_true, y_pred, average='weighted')
    F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    F05 = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ACC = metrics.accuracy_score(y_true, y_pred)
    print(f'''Top-1 (Greedy Decoding)
            Precision: {PR:.4f}
            Recall: {RE:.4f}
            F1 Score: {F1:.4f}
            F0.5 Score: {F05:.4f}
            Accuracy: {ACC * 100:.2f}%
            Modified Accuracy: {modified_acc * 100:.2f}%
    ''')

    return evaluation_df

    # evaluation_df.sample(10)
