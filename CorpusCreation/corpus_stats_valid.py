from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import time
import pandas as pd
import re
import sys
import argparse
from tqdm import tqdm


# ########################################################
def login():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="Enter Your Email")
    parser.add_argument("--password", help="Enter Your Facebook Password")
    args = parser.parse_args()

    # code to ignore browser notifications
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome('./chromedriver.exe', chrome_options=chrome_options)
    # open the webpage
    driver.get("https://wwww.facebook.com/")
    # target username
    username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='email']")))
    password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='pass']")))
    # entering email as username
    username.clear()
    username.send_keys(args.email)
    # entering password
    password.clear()
    password.send_keys(args.password)
    # target the login button and click it
    time.sleep(5)
    button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()
    # We are logged in!
    print("Logged in")
    return driver
# ########################################################


# ########################################################
def scrape_post_1():
    driver = login()
    # https://fb.watch/eN-nBOb45t/
    url = "https://mbasic.facebook.com/story.php?story_fbid=pfbid02TjtvmwDs51fyVRaHbvM5XgxL1gBGb6USBYvsxgMdn8c4BcQvjbLv1BFCjw52UsXQl&id=111762869482599&eav=Afba2OolCuRXElnzf97xViXfIosR66LZPdko_Q9oxtd5fhvZMDjeKOC_JD1Nx2LKtEE&__tn__=%2AW&paipv=0"
    # url = "https://mbasic.facebook.com/story.php?story_fbid=pfbid0eP3VufmYZQEdDrGybgzg9ganLPXRo9JXQ8q5pUjiaBF7gTQ9FnkJdw44PDfx11JKl&id=313147292549612&eav=AfbiujhhnbU2KOwEYD6oavgC5llyK5uWWqiecav3DYpPCCC4llyMqpaYY9rPUvap1z0&ref=sharing&__tn__=%2AW&paipv=0"
    while True:
        driver.get(url)
        comments = driver.find_element(By.CLASS_NAME, "ef").text
        comments = re.sub("[A-Za-z0-9·\\n]", "", comments)
        next_page = driver.find_elements(By.TAG_NAME, "a")[-1].get_attribute('href')
        if type(next_page) != str:
            break
        url = next_page
        time.sleep(5)
        sys.exit()
        with open('./dfs/comments.txt', 'a', encoding='utf-8') as f:
            f.write(comments)
            f.write(' \n ')
# ########################################################


# ########################################################
def scrape_post_2():
    driver = login()
    # https://fb.watch/eNQHYjDuA6/
    url = "https://mbasic.facebook.com/story.php?story_fbid=pfbid0eP3VufmYZQEdDrGybgzg9ganLPXRo9JXQ8q5pUjiaBF7gTQ9FnkJdw44PDfx11JKl&id=313147292549612&eav=AfbiujhhnbU2KOwEYD6oavgC5llyK5uWWqiecav3DYpPCCC4llyMqpaYY9rPUvap1z0&ref=sharing&__tn__=%2AW&paipv=0"
    while True:
        driver.get(url)
        comments = driver.find_elements(By.CLASS_NAME, "eb")
        for comment in comments:
            comment = comment.text
            comment = re.sub("[A-Za-z0-9·.\\n]", "", comment)
            with open('comments.txt', 'a', encoding='utf-8') as f:
                f.write(comment)
                f.write('  ')

        comments = driver.find_elements(By.CLASS_NAME, "ec")
        for comment in comments:
            comment = comment.text
            comment = re.sub("[A-Za-z0-9·.\\n]", "", comment)
            with open('./dfs/comments.txt', 'a', encoding='utf-8') as f:
                f.write(comment)
                f.write('  ')

        next_page = driver.find_elements(By.TAG_NAME, "a")[-1].get_attribute('href')
        if type(next_page) != str:
            break

        url = next_page
        time.sleep(5)
# ########################################################


# ########################################################
def clean_text(text):
    all_chars = ['ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
                 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ',
                 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ',
                 'ষ', 'স', 'হ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ',
                 'ৗ', 'ড়', 'ঢ়', 'য়']
    cleaned_text = ''
    for i in tqdm(range(len(text))):
        if text[i] in all_chars:
            cleaned_text += text[i]
        else:
            cleaned_text += ' '
    return cleaned_text

def find_stats():
    f = open("./dfs/comments.txt", "r", encoding='utf-8')
    text = f.read()
    text = clean_text(text)

    words = sorted(text.split())
    unique_words = sorted(list(set(words)))

    error_df = pd.read_csv('./dfs/sec_dataset_IV.csv')
    balanced_df = pd.DataFrame()
    all_error_types = sorted(list(set(error_df.iloc[:, -1].values)))
    for error in all_error_types:
        x = error_df.loc[error_df['ErrorType'] == error]
        if (len(x)) < 100000:
            balanced_df = pd.concat([balanced_df, x])
        else:
            balanced_df = pd.concat([balanced_df, x.sample(100000)])

    erroneous_words = balanced_df.iloc[:, 1].values
    erroneous_words_type = balanced_df.iloc[:, 2].values

    found = []
    types = []
    for i in tqdm(range(len(unique_words))):
        word = unique_words[i]
        if word in erroneous_words:
            found.append(word)
            types.append(erroneous_words_type[i])
        if (i != 0 and i % 1000 == 0):
            print(len(found))

    error_words = []
    error_types = []
    for i in tqdm(range(len(found))):
        word = found[i]
        etype = error_df.loc[error_df['Error'] == word]['ErrorType'].values[0]
        error_words.append(word)
        error_types.append(etype)

    temp = pd.DataFrame({
        'Error': error_words,
        'ErrorType': error_types
    })

    unique_etypes = sorted(list(set(error_types)))
    err_names, instances, pcts = [], [], []
    for etype in unique_etypes:
        x = temp.loc[temp['ErrorType'] == etype]
        print(f"{etype}, {len(x)}/{len(temp)}, {len(x) / len(temp) * 100:.2f}%")
        err_names.append(etype)
        instances.append(f"{len(x)}/{len(temp)}")
        pcts.append(len(x) / len(temp) * 100)

    df = pd.DataFrame({
        'ErrorType': err_names,
        'Instances': instances,
        'Pct': pcts
    })
    print(df)

    print("Missing error types")
    found = sorted(list(set(error_types)))
    target = sorted(list(set(error_df.iloc[:, -1].values)))

    for item in target:
        if item not in found:
            print(item)
# ########################################################


# ########################################################
if __name__ == '__main__':
    scrape_post_1()
    scrape_post_2()
    find_stats()
