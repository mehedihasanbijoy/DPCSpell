import pandas as pd
from tqdm import tqdm
from pipeline import translate_sentence
import numpy as np
from sklearn import metrics
import torch
import gc
import warnings as wrn
wrn.filterwarnings('ignore')


def evaluation_report(test_data, SRC, TRG, model, DEVICE):
    erroneous_words, predicted_words, correct_words, flags = [], [], [], []

    modified_flags = []
    all_words = pd.read_csv('./Dataset/allDictWords_df.csv')
    all_words = sorted(all_words.iloc[:, 0].values)

    for idx, data in enumerate(tqdm(test_data)):
        # ------------------------------
        if idx % 5000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # ------------------------------

        src = data.src
        trg = data.trg
        translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)

        src = ''.join(src)
        trg = ''.join(trg)
        pred = ''.join(translation)

        erroneous_words.append(src)
        correct_words.append(trg)
        predicted_words.append(pred)

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

    return evaluation_df


def evaluation_report2(test_data, SRC, TRG, WORD, model, DEVICE):
    erroneous_words, predicted_words, correct_words, flags = [], [], [], []
    words = []

    modified_flags = []
    all_words = pd.read_csv('./Dataset/allDictWords_df.csv')
    all_words = sorted(all_words.iloc[:, 0].values)

    for idx, data in enumerate(tqdm(test_data)):
        # ------------------------------
        if idx % 5000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # ------------------------------

        src = data.src
        trg = data.trg
        word = data.word
        translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)

        src = ''.join(src)
        trg = ''.join(trg)
        pred = ''.join(translation)
        word = ''.join(word)

        erroneous_words.append(src)
        correct_words.append(trg)
        predicted_words.append(pred)
        words.append(word)

        if trg == pred:
            flags.append(1)
        else:
            flags.append(0)

        if pred in all_words:
            modified_flags.append(1)
        else:
            modified_flags.append(0)

    evaluation_df = pd.DataFrame({
        'Error': erroneous_words,  # Error
        'Predicton': predicted_words,  # ErrorBlanksPredD1
        'Target': correct_words,  # ErrorBlanksActual
        'Word': words,  # Word
        'Correction': flags  # EBP_Flag_D1
    })

    corrected_instances = evaluation_df['Correction'].values.sum()
    total_instances = len(evaluation_df)
    accuracy = corrected_instances / total_instances
    print(f"\nCorrection/Total: {corrected_instances} / {total_instances}")

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

    return evaluation_df



def evaluation_report3(test_data, SRC, TRG, ERROR, WORD, EBPD1, EBPFD1, model, DEVICE):
    erroneous_words, predicted_words, correct_words, flags = [], [], [], []
    errors = []
    words = []
    ebpd1s = []
    ebpfd1s = []

    modified_flags = []
    all_words = pd.read_csv('./Dataset/allDictWords_df.csv')
    all_words = sorted(all_words.iloc[:, 0].values)

    for idx, data in enumerate(tqdm(test_data)):
        # ------------------------------
        if idx % 5000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # ------------------------------

        src = data.src
        trg = data.trg
        error = data.error
        word = data.word
        ebpd1 = data.ebpd1
        ebpfd1 = data.ebpfd1

        translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)

        src = ''.join(src)
        trg = ''.join(trg)
        pred = ''.join(translation)
        error = ''.join(error)
        word = ''.join(word)
        ebpd1 = ''.join(ebpd1)
        ebpfd1 = ''.join(ebpfd1)

        erroneous_words.append(src)
        correct_words.append(trg)
        predicted_words.append(pred)
        errors.append(error)
        words.append(word)
        ebpd1s.append(ebpd1)
        ebpfd1s.append(ebpfd1)

        if trg == pred:
            flags.append(1)
        else:
            flags.append(0)

        if pred in all_words:
            modified_flags.append(1)
        else:
            modified_flags.append(0)

    # evaluation_df = pd.DataFrame({
    #     'Error': erroneous_words,
    #     'Predicton': predicted_words,
    #     'Target': correct_words,
    #     'Word': words,
    #     'Correction': flags
    # })

    evaluation_df = pd.DataFrame({
        'Error': errors,
        'Word': words,
        'ErrorBlanksActual': correct_words,
        'MaskErrorBlank': erroneous_words,
        'ErrorBlanksPredD1': ebpd1s,
        'EBP_Flag_D1': ebpfd1s,
        'ErrorBlanksPredD2': predicted_words,
        'EBP_Flag_D2': flags
    })

    corrected_instances = evaluation_df['EBP_Flag_D2'].values.sum()
    total_instances = len(evaluation_df)
    accuracy = corrected_instances / total_instances
    print(f"\nCorrection/Total: {corrected_instances} / {total_instances}")

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

    return evaluation_df




if __name__ == '__main__':
    pass


