import requests, bs4
import pandas as pd
from tqdm import tqdm


def word_accumulation():
    char_pages = {
        'অ': 71, 'আ': 50, 'ই': 10, 'ঈ': 1, 'উ': 25, 'ঊ': 2, 'ঋ': 1, 'এ': 13, 'ঐ': 2, 'ও': 7, 'ঔ': 3,
        'ক': 82, 'খ': 29, 'গ': 35, 'ঘ': 7, 'ঙ': 1, 'চ': 32, 'ছ': 12, 'জ': 28, 'ঝ': 8, 'ঞ': 1,
        'ট': 16, 'ঠ': 4, 'ড': 12, 'ঢ': 6, 'ণ': 1, 'ত': 44, 'থ': 6, 'দ': 44, 'ধ': 13, 'ন': 52,
        'প': 77, 'ফ': 16, 'ব': 90, 'ভ': 24, 'ম': 58, 'য': 11, 'র': 30, 'ল': 18, 'শ': 25, 'ষ': 3, 'স': 86, 'হ': 27
    }

    all_urls = {}

    url = 'https://accessibledictionary.gov.bd/bengali-to-bengali/'

    html_codes = requests.get(url).text
    document = bs4.BeautifulSoup(html_codes, 'lxml')
    alphabet_links = document.find('ul', class_='alphabet')
    items = alphabet_links.find_all('li')

    for item in items:
        url = str(item).split('"')[1]
        all_urls[url[-1:]] = url

    df_dict = {}

    for url in all_urls.values():
        no_of_pages = char_pages[url[-1:]]
        for idx in tqdm(range(1, no_of_pages + 1)):
            desired_url = url + '&page=' + str(idx)
            html_codes = requests.get(desired_url).text
            document = bs4.BeautifulSoup(html_codes, 'lxml')
            article = document.find('article', class_='dicDisplay')
            items = article.find_all('li')

            for item in items:
                text = item.get_text()
                text = text.split('Bengali Word')[1]
                text = text.split('Bengali definition')
                ben_word = text[0]
                ben_def = text[1]
                df_dict[ben_word] = ben_def
            # break

    df = pd.DataFrame(
        {
            'Word': df_dict.keys(),
            'Defination': df_dict.values()
        }
    )
    return df


def get_len(word):
    return len(word)


def text_preprocessing(df):
    all_chars = ['ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
                 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ',
                 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ',
                 'ষ', 'স', 'হ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ',
                 'ৗ', 'ড়', 'ঢ়', 'য়', ' ']

    words = ''

    df_words = ' '.join(df['Word'].values)
    for char in df_words:
        if char in all_chars:
            words += char

    words += ' '

    df_definations = ' '.join(df['Defination'].values)
    for char in df_definations:
        if char in all_chars:
            words += char

    words = sorted(list(set(words.split(' '))))
    df_all_words = pd.DataFrame({'word': words})
    df_all_words['len'] = df_all_words['word'].apply(get_len)
    df_all_words = df_all_words.loc[df_all_words['len'] > 2]
    return df_all_words


if __name__ == '__main__':
    df = word_accumulation()
    df_all_words = text_preprocessing(df)
    df_all_words.to_csv('./dfs/df_all_words.csv', index=False)
