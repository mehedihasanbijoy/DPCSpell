import pandas as pd
from utils import word2char


def error_df(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    df.to_csv('./Dataset/error.csv', index=False)

