from utils import translate_sentence

from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import numpy as np


def evaluation_report(test_data, SRC, TRG, model, DEVICE):
    erroneous_words, predicted_words, correct_words, flags = [], [], [], []

    modified_flags = []
    all_words = pd.read_csv('./Dataset/allDictWords_df.csv')
    all_words = sorted(all_words.iloc[:, 0].values)

    for data in tqdm(test_data):
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

    evaluation_df = pd.DataFrame({
        'Error': erroneous_words,
        'Predicton': predicted_words,
        'Target': correct_words,
        'Correction': flags
    })

    corrected_instances = evaluation_df['Correction'].values.sum()
    total_instances = len(evaluation_df)
    accuracy = corrected_instances / total_instances
    print(f"\nCorrection/Total: {corrected_instances} / {total_instances}")
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    y_true = np.array(correct_words)
    y_pred = np.array(predicted_words)

    PR = metrics.precision_score(y_true, y_pred, average='weighted')
    RE = metrics.recall_score(y_true, y_pred, average='weighted')
    F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    F05 = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ACC = metrics.accuracy_score(y_true, y_pred)
    MODIFIED_ACC = np.sum(modified_flags) / len(modified_flags)

    print(f'''
    Top-1 (Greedy Decoding)
        Precision: {PR:.4f}
        Recall: {RE:.4f}
        F1 Score: {F1:.4f}
        F0.5 Score: {F05:.4f}
        Accuracy: {RE * 100:.2f}%
        Modified Accuracy: {MODIFIED_ACC * 100:.2f}%
    ''')

    # evaluation_df.to_csv('./Dataset/preds_convs2s.csv', index=False)
    return evaluation_df


if __name__ == '__main__':
    pass