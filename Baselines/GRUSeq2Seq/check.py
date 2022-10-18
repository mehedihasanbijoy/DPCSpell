import pandas as pd
import numpy as np
from tqdm import tqdm


def within_topk(df, k):
    correct = df['Correct']
    topk = df.iloc[:, 1:k+1].values
    preds = 0
    # for idx in tqdm(range(len(df))):
    for idx in range(len(df)):
        if correct[idx] in topk[idx]:
            preds += 1
    acc_within_topk = preds / len(df)
    print(f"Within Top-{k} Acc: {acc_within_topk}")


def modified_acc(df_allWords, df, k):
    df_allWords = sorted(df_allWords.iloc[:, 0].values)
    correct = df['Correct']
    topk = df.iloc[:, 1:k + 1].values
    preds = 0
    for words in tqdm(topk):
        for word in words:
            if word in df_allWords:
                preds += 1
                break;
    modified_acc_within_topk = preds / len(df)
    print(f"Within Top-{k} Modified Acc: {modified_acc_within_topk}")


def beam_report():
    print("""
    --------------------------------
    Beam Decoding Evaluation Report
    --------------------------------
    """)
    df_allWords = pd.read_csv('./Dataset/allDictWords_df.csv')
    df_beam = pd.read_csv('./Corrections/preds_beam_colab.csv')
    top1_acc = np.sum(df_beam['Pred-1'] == df_beam['Correct']) / len(df_beam)
    top2_acc = np.sum(df_beam['Pred-2'] == df_beam['Correct']) / len(df_beam)
    top3_acc = np.sum(df_beam['Pred-3'] == df_beam['Correct']) / len(df_beam)
    print(f"Top1 Acc: {top1_acc}")
    print(f"Top2 Acc: {top2_acc}")
    print(f"Top3 Acc: {top3_acc}\n")
    within_topk(df_beam, 1)
    within_topk(df_beam, 2)
    within_topk(df_beam, 3)
    modified_acc(df_allWords, df_beam, 1)
    modified_acc(df_allWords, df_beam, 2)
    modified_acc(df_allWords, df_beam, 3)

def test():
    df = pd.read_csv('./Dataset/allDictWords_df.csv')
    words = sorted(df.iloc[:, 0].values)
    print(words)
#
# acc = (df_beam['Pred-1'] == df_beam['Correct'])*1 + \
#         (df_beam['Pred-2'] == df_beam['Correct'])*1 + \
#         (df_beam['Pred-3'] == df_beam['Correct'])*1
# acc = acc.values
# acc = [1 if x>0 else 0 for x in acc]
# print(f"Accuracy: {np.sum(acc) / len(df_beam)}")
#
# df_dict = pd.read_csv('./Dataset/allDictWords_df.csv')
# df_allWords = pd.read_csv('./Dataset/df_all_words.csv')
# #
# preds1 = []
# for word in tqdm(df_beam['Pred-1'].values):
#     # similar_words = df_dict.loc[df_dict['word'].str.startswith(word)].iloc[:, 0].values
#     if word in df_allWords.iloc[:, 0].values:
#         preds1.append(1)
#     else:
#         preds1.append(0)
# print(f"Modified Top1 Acc: {np.sum(preds1) / len(preds1)}")
#
# df_greedy = pd.read_csv('./Corrections/preds_greedy_colab.csv')
# # print(df_greedy)
# greedy_acc = np.sum(df_greedy['Predicton'] == df_greedy['Target'])/len(df_greedy)
# print(f'Greedy Accuracy: {greedy_acc}')
# preds = []
# for word in tqdm(df_greedy['Predicton'].values):
#     if word in df_allWords.iloc[:, 0].values:
#         preds.append(1)
#     else:
#         preds.append(0)
# print(f"Modified Greedy Accuracy: {np.sum(preds) / len(preds)}")

if __name__ == '__main__':
    beam_report()