import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

import warnings as wrn
wrn.filterwarnings('ignore')

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# ---------------------------
def train_valid_test_df(df, test_size, valid_size):
    # etypes = list(set(df.iloc[:, -1]))
    etypes = list(set(df['ErrorType']))

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for etype in etypes:
        etype_df = df.loc[df['ErrorType'] == etype]
        train, test = train_test_split(etype_df, test_size=test_size)
        train, valid = train_test_split(train, test_size=valid_size)

        train_df = pd.concat([train_df, train])
        valid_df = pd.concat([valid_df, valid])
        test_df = pd.concat([test_df, test])

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    train_df = train_df.iloc[:, [1, 0]]
    valid_df = valid_df.iloc[:, [1, 0]]
    test_df = test_df.iloc[:, [1, 0]]

    return train_df, valid_df, test_df
# ---------------------------


# ---------------------------
def train_valid_test_df2(df, test_size, valid_size):
    # etypes = list(set(df.iloc[:, -1]))
    etypes = list(set(df['ErrorType']))

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for etype in etypes:
        etype_df = df.loc[df['ErrorType'] == etype]
        train, test = train_test_split(etype_df, test_size=test_size)
        train, valid = train_test_split(train, test_size=valid_size)

        train_df = pd.concat([train_df, train])
        valid_df = pd.concat([valid_df, valid])
        test_df = pd.concat([test_df, test])

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # train_df = train_df.iloc[:, [1, 0]]
    # valid_df = valid_df.iloc[:, [1, 0]]
    # test_df = test_df.iloc[:, [1, 0]]

    return train_df, valid_df, test_df
# ---------------------------


# ---------------------------
def merge_dfs(network='detector'):
    df_names = [
        f'{network}_CognitiveError.csv',
        f'{network}_HomonymError.csv',
        f'{network}_Run-onError.csv',
        f'{network}_Split-wordErrorLeft.csv',
        f'{network}_Split-wordErrorRandom.csv',
        f'{network}_Split-wordErrorRight.csv',
        f'{network}_Split-wordErrorboth.csv',
        f'{network}_TypoAvroSubstituition.csv',
        f'{network}_TypoBijoySubstituition.csv',
        f'{network}_TypoDeletion.csv',
        f'{network}_TypoInsertion.csv',
        f'{network}_TypoTransposition.csv',
        f'{network}_VisualError.csv',
        f'{network}_VisualErrorCombinedCharacter.csv'
    ]
    
    df = pd.DataFrame()

    for df_name in df_names:
        df_path = os.path.join('./Dataframes', df_name)
        temp_df = pd.read_csv(df_path)
        temp_df['ErrorType'] = [df_name.split('.')[0].split('_')[-1]
                                for _ in range(len(temp_df))]
        df = pd.concat([df, temp_df])

    df = df.iloc[:, :]

    if network=='detector':
        df.rename(
            columns = {
                'Predicton':'ErrorBlanksPredD1', 
                'Target':'ErrorBlanksActual', 
                'Correction':'EBP_Flag_D1', 
            }, 
            inplace = True
        )
        df = df[['Error', 'Word', 'ErrorBlanksPredD1', 'ErrorBlanksActual', 'EBP_Flag_D1', 'ErrorType']]

    df.to_csv(f'./Dataset/{network}_preds.csv', index=False)  # sec_dataset_III_v3_masked_d1_gen.csv (detector)
                                                               # (purificator)
# ---------------------------


# ---------------------------
def error_df(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    idx = int(len(df)/1)
    df = df.iloc[:idx, [1, 0]]
    df.to_csv('./Dataset/error.csv', index=False)
# ---------------------------


# ---------------------------
def error_df_2(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    # df['Word'] = df['Word'].apply(word2char)
    # df['MaskErrorBlank'] = df['MaskErrorBlank'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    idx = int(len(df)/1)
    df = df.iloc[:idx, [1, 0]]
    #
    # if(len(df) >= 10000):
    #     df = df.iloc[:10000, :]
    #
    df.to_csv('./Dataset/error.csv', index=False)
# ---------------------------


# ---------------------------
def error_df_3(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    # df['Word'] = df['Word'].apply(word2char)
    # df['MaskErrorBlank'] = df['MaskErrorBlank'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    # idx = int(len(df)/1)
    # df = df.iloc[:idx, [1, 0]]
    #
    # if(len(df) >= 10000):
    #     df = df.iloc[:10000, :]
    #
    df.to_csv('./Dataset/error.csv', index=False)
# ---------------------------


# ---------------------------
def word2char(word):
    w2c = [char for char in word]
    return ' '.join(w2c)
# ---------------------------


# ---------------------------
def find_len(seq):
    return len(seq.split(' '))
# ---------------------------


# ---------------------------
def mask2str(mask):
    x = ''
    for item in mask:
        if item != "[" and item != "'" and item != "," and item != " " and item != "]":
            x += str(item)
    return x
# ---------------------------


# ---------------------------
def error_blank(error, mask):
    error_list = np.array(error.split())
    mask_list = np.array(mask.split())
    idx = np.where(mask_list=='1')[0]
    error_list[idx] = ' '
    error = ' '.join(error_list)
    return error
# ---------------------------


# ---------------------------
def basic_tokenizer(text):
    return text.split()
# ---------------------------


# ---------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# ---------------------------


# ---------------------------
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
# ---------------------------


# ---------------------------
def save_model(model, train_loss, epoch, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, PATH)
    print(f"---------\nModel Saved at {PATH}\n---------\n")
# ---------------------------


# ---------------------------
def load_model(model, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    return checkpoint, epoch, train_loss
# ---------------------------


if __name__ == '__main__':
    pass
