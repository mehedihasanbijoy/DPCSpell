import pandas as pd
from utils import word2char
from tqdm import tqdm


def check_from_left(word, error):
    left = []
    for i in range(len(error)):
        if error[i] == word[i]:
            left.append(0)
        else:
            left.append(1)
    return left


def check_from_right(word, error):
    word.reverse()
    error.reverse()
    right = []
    for i in range(len(error)):
        if error[i] == word[i]:
            right.append(0)
        else:
            right.append(1)
    right.reverse()
    return right


def check_from_both(word, error):
    length = len(error)
    if length % 2 == 0:
        iterator = length // 2
    else:
        iterator = (length // 2) + 1

    x = -1

    left = []
    right = []

    for i in range(iterator):
        if error[i] == word[i]:
            left.append(0)
        else:
            left.append(1)

        if error[x] == word[x]:
            right.append(0)
        else:
            right.append(1)
        x -= 1

    right.reverse()
    both = [*left, *right]
    return both


if __name__ == '__main__':
    path = './Dataset/sec_dataset_III_v3.csv'
    df = pd.read_csv('./Dataset/sec_dataset_III_v3.csv')
    df_copy = df.copy()
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)

    for idx in tqdm(range(len(df))):
        word = df.iloc[idx, 0].split()
        error = df.iloc[idx, 1].split()
        word = ['ব', 'া', 'ং', 'ল', 'া']
        error = ['ব', 'ং', 'ল', 'া']
        print(len(word), len(error))
        print(f'{word}\n{error}')
        # checking from left
        left = check_from_left(word, error)
        print(left)
        right = check_from_right(word, error)
        print(right)
        both = check_from_both(word, error)
        print(both)
        break
