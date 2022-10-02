import math
import random
import pandas as pd
from tqdm import tqdm


# ###########################################################
def generate_erroneous_words(df, error_dict):
    actual_words = []
    erroneous_word = []

    single_error = []

    words = df.iloc[:, 0].values

    for i in tqdm(range(len(words))):
        word = words[i]

        word2list = [x for x in word]
        potential_candidates = []

        for char in word2list:
            if char in list(error_dict.keys()):
                potential_candidates.append(char)

        size = len(potential_candidates)

        if (size == 0):
            continue
        else:
            for i in range(math.ceil(size / size)):
                target_key = random.choice(potential_candidates)
                try:
                    word2list[word2list.index(target_key)] = random.choice(error_dict[target_key])
                except:
                    try:
                        word2list[word2list.index(target_key)] = random.choice(error_dict[target_key])
                    except:
                        pass
            transformed_word = ''.join(word2list)
            if transformed_word not in df.iloc[:, 0].values:
                actual_words.append(word)
                erroneous_word.append(transformed_word)

                if (len(transformed_word) == len(word)):
                    single_error.append(1)
                else:
                    single_error.append(0)

    return actual_words, erroneous_word, single_error
# ###########################################################


# ###########################################################
def cognitive_phonetic_error(df):
    dict_phoneticError = {
        'ই': ['ঈ'],
        'ঈ': ['ই'],
        'উ': ['ঊ'],
        'ঊ': ['উ'],
        'চ': ['ছ'],
        'ছ': ['চ'],
        'জ': ['ঝ', 'য'],
        'ঝ': ['জ', 'য'],
        'ড': ['ঢ'],
        'ঢ': ['ড'],
        'ণ': ['ন'],
        'দ': ['ধ'],
        'ধ': ['দ'],
        'ন': ['ণ'],
        'প': ['ফ'],
        'ফ': ['প'],
        'ব': ['ভ'],
        'ভ': ['ব'],
        'য': ['জ', 'ঝ'],
        'শ': ['ষ', 'স'],
        'ষ': ['শ', 'স'],
        'স': ['ষ', 'শ']
    }

    dfs_cognitive_phonetic = pd.DataFrame()

    for i in range(5):
        phonetic_actual, phonetic_error, single_error = generate_erroneous_words(df, dict_phoneticError)
        df_cognitive_phonetic = pd.DataFrame(
            {
                'Word': phonetic_actual,
                'Error': phonetic_error,
                'ErrorType': ['Cognitive Error' for x in range(len(phonetic_error))],
                'SingleError': single_error
            }
        )
        dfs_cognitive_phonetic = pd.concat([dfs_cognitive_phonetic, df_cognitive_phonetic])

    dfs_cognitive_phonetic = dfs_cognitive_phonetic.drop_duplicates(subset=['Error'])
    return dfs_cognitive_phonetic
# ###########################################################


# ###########################################################
def visual_error(df):
    dict_visualError = {
        'ঃ': [':', ';'],
        'অ': ['আ'],
        'আ': ['অ'],
        'ই': ['ঈ'],
        'ঈ': ['ই'],
        'উ': ['ঊ', 'ড', ],
        'ঊ': ['উ'],
        'ঋ': ['ঝ'],
        'এ': ['ত্র'],
        'ঐ': ['এ'],
        'ও': ['ত্ত', ],
        'ঔ': ['ও', 'ঐ'],
        'ক': ['ব'],
        'খ': ['ঋ'],
        'ঘ': ['য', 'ষ'],
        'ট': ['ঢ'],
        'ড': ['ড়'],
        'ঢ': ['ঢ়'],
        'ণ': ['ন'],
        'ন': ['ণ'],
        'ব': ['র'],
        'য': ['য়', 'ষ'],
        'র': ['ব'],
        'ল': ['ন'],
        'ষ': ['য', 'য়'],
        'ী': ['ৗ'],
        'ু': ['ূ'],
        'ূ': ['ু', 'ৃ'],
        'ৃ': ['ূ'],
        'ৗ': ['ী'],
        'ড়': ['ড'],
        'ঢ়': ['ঢ'],
        'য়': ['য', 'ষ']
    }

    dfs_visual = pd.DataFrame()

    for i in range(5):
        visual_actual, visual_error, single_error = generate_erroneous_words(df, dict_visualError)
        df_visual = pd.DataFrame(
            {
                'Word': visual_actual,
                'Error': visual_error,
                'ErrorType': ['Visual Error' for x in range(len(visual_error))],
                'SingleError': single_error
            }
        )
        dfs_visual = pd.concat([dfs_visual, df_visual])

    dfs_visual = dfs_visual.drop_duplicates(subset=['Error'])
    return dfs_visual
# ###########################################################


# ###########################################################
def filter_combined_characters(word):
    temp = [x for x in word]
    target = []
    for i in range(len(temp)):
        if temp[i] != '্':
            if i+1 <= len(temp)-2 and temp[i+1] == '্':
                target.append(''.join([temp[i], temp[i+1], temp[i+2]]))
                i += 3
            else:
                if len(target) == 0:
                    target.append(temp[i])
                else:
                    if temp[i] != [x for x in target[-1]][-1]:
                        target.append(temp[i])
        elif temp[i] == '্':
            i += 2
    return target
# ###########################################################


# ###########################################################
def generate_erroneous_words_(df, error_dict):
    actual_words = []
    erroneous_word = []
    single_error = []

    words = df['filtered'].values

    for i in tqdm(range(len(words))):
        word = words[i]

        word2list = filter_combined_characters(word)
        potential_candidates = []

        for char in word2list:
            if char in list(error_dict.keys()):
                potential_candidates.append(char)

        size = len(potential_candidates)

        if (size == 0):
            continue
        else:
            for i in range(math.ceil(size / size)):
                target_key = random.choice(potential_candidates)
                try:
                    word2list[word2list.index(target_key)] = random.choice(error_dict[target_key])
                except:
                    try:
                        word2list[word2list.index(target_key)] = random.choice(error_dict[target_key])
                    except:
                        pass
            if ''.join(word2list) not in df.iloc[:, 0].values:
                actual_words.append(''.join(word))
                erroneous_word.append(''.join(word2list))

                if (len(word) == len(word2list)):
                    single_error.append(1)
                else:
                    single_error.append(0)

    return actual_words, erroneous_word, single_error
# ###########################################################


# ###########################################################
def visual_error_combined_char(df):
    df['filtered'] = df['word'].apply(filter_combined_characters)
    dict_combined = {
        'এ': ['ত্র'],
        'ও': ['ত্ত'],
        'ত্ত': ['ও'],
        'ক্ত': ['ত্ত'],
        'ক্ম': ['ক্স'],
        'ক্ষ': ['হ্ম'],
        'ক্স': ['ক্ম'],
        'গু': ['ণ্ড'],
        'চ্চ': ['চ্চ'],
        'চ্ছ': ['চ্ছ'],
        'ণ্ড': ['গু'],
        'ণ্ঢ': ['ন্ট'],
        'ত্র': ['এ'],
        'দ্ধ': ['দ্ব'],
        'দ্ব': ['দ্ধ'],
        'ন্ট': ['ণ্ঢ', 'ল্ট'],
        'ন্স': ['প্স'],
        'প্ন': ['প্ল'],
        'প্ল': ['প্ন'],
        'প্স': ['ন্স'],
        'ল্ট': ['ন্ট'],
        'শ্চ': ['শ্ছ'],
        'শ্ছ': ['শ্চ'],
        'ষ্ক': ['স্ক'],
        'ষ্ট': ['স্ট'],
        'স্ক': ['ষ্ক'],
        'স্ট': ['ষ্ট'],
        'হ্ম': ['ক্ষ']
    }

    dfs_combinedchar_visual_error = pd.DataFrame()

    for i in range(5):
        combined_actual, combined_error, single_error = generate_erroneous_words_(df, dict_combined)
        df_combinedchar_visual_error = pd.DataFrame(
            {
                'Word': combined_actual,
                'Error': combined_error,
                'ErrorType': ['Visual Error (Combined Character)' for x in range(len(combined_error))],
                'SingleError': single_error
            }
        )
        dfs_combinedchar_visual_error = pd.concat([dfs_combinedchar_visual_error, df_combinedchar_visual_error])

    dfs_combinedchar_visual_error = dfs_combinedchar_visual_error.drop_duplicates(subset=['Error'])
    return dfs_combinedchar_visual_error
# ###########################################################


# ###########################################################
def typo_insertion(word):
    unique_chars = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
                    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট',
                    'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব',
                    'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়']

    word2chars = [x for x in word]
    while True:
        target_char = random.choice(word2chars)
        # idx = word2chars.index(target_char)
        # if idx <= len(word2chars)-2 and target_char in unique_chars and word2chars[idx+1] in unique_chars:
        if target_char in unique_chars:
            break
    target_char_idx = word2chars.index(target_char)
    word2chars.insert(target_char_idx+1, target_char)
    return ''.join(word2chars)
# ###########################################################


# ###########################################################
def typo_insertion_error(df):
    dfs_typo_insertion_error = pd.DataFrame()
    for i in range(5):
        typoInsertion_actual, typoInsertion_error = [], []

        single_error = []

        words = df.iloc[:, 0].values

        for i in tqdm(range(len(words))):
            word = words[i]
            transformed_word = typo_insertion(word)
            if transformed_word not in df.iloc[:, 0].values:
                typoInsertion_actual.append(word)
                typoInsertion_error.append(transformed_word)

                if (len(typoInsertion_actual) == len(typoInsertion_error)):
                    single_error.append(1)
                else:
                    single_error.append(0)

        df_typo_insertion_error = pd.DataFrame(
            {
                'Word': typoInsertion_actual,
                'Error': typoInsertion_error,
                'ErrorType': ['Typo Insertion' for x in range(len(typoInsertion_error))],
                'SingleError': single_error
            }
        )

        dfs_typo_insertion_error = pd.concat([dfs_typo_insertion_error, df_typo_insertion_error])

    dfs_typo_insertion_error = dfs_typo_insertion_error.drop_duplicates(subset=['Error'])
    return dfs_typo_insertion_error
# ###########################################################


# ###########################################################
def typo_deletion_error(df):
    all_chars = ['ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
                 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ',
                 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ',
                 'ষ', 'স', 'হ', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ',
                 'ৗ', 'ড়', 'ঢ়', 'য়']

    dfs_typo_deletion_error = pd.DataFrame()
    for i in range(5):
        typoDeletion_actual, typoDeletion_error = [], []
        single_error = []

        words = df.iloc[:, 0].values
        for i in tqdm(range(len(words))):
            word = words[i]
            word2chars = [x for x in word]
            while True:
                target_char = random.choice(word2chars)
                if target_char in all_chars:
                    break
            word2chars.remove(target_char)
            transformed_word = ''.join(word2chars)
            if transformed_word in df.iloc[:, 0].values:
                continue
            else:
                typoDeletion_actual.append(word)
                typoDeletion_error.append(transformed_word)

                if (len(typoDeletion_actual) == len(typoDeletion_actual)):
                    single_error.append(1)
                else:
                    single_error.append(0)

        df_typo_deletion_error = pd.DataFrame(
            {
                'Word': typoDeletion_actual,
                'Error': typoDeletion_error,
                'ErrorType': ['Typo Deletion' for x in range(len(typoDeletion_error))],
                'SingleError': single_error
            }
        )
        dfs_typo_deletion_error = pd.concat([dfs_typo_deletion_error, df_typo_deletion_error])

    dfs_typo_deletion_error = dfs_typo_deletion_error.drop_duplicates(subset=['Error'])
    return dfs_typo_deletion_error
# ###########################################################


# ###########################################################
def typo_transposition(word):
    unique_chars = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ',
                    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট',
                    'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব',
                    'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়']

    word2chars = [x for x in word]
    while True:
        target_char = random.choice(word2chars)
        # if target_char in all_chars:
        if target_char in unique_chars:
            idx = word2chars.index(target_char)
            if idx <= len(word2chars) - 2:
                break
    word2chars[idx], word2chars[idx+1] = word2chars[idx+1], word2chars[idx]
    return ''.join(word2chars)
# ###########################################################


# ###########################################################
def typo_transposition_error(df):
    dfs_typo_transposition_error = pd.DataFrame()
    for i in range(5):
        typoTransposition_actual, typoTransposition_error = [], []
        single_error = []

        words = df.iloc[:, 0].values
        for i in tqdm(range(len(words))):
            word = words[i]
            transformed_word = typo_transposition(word)
            if transformed_word in df.iloc[:, 0].values:
                continue

            typoTransposition_actual.append(word)
            typoTransposition_error.append(transformed_word)

            if (len(typoTransposition_actual) == len(typoTransposition_error)):
                single_error.append(1)
            else:
                single_error.append(0)

        df_typo_transposition_error = pd.DataFrame(
            {
                'Word': typoTransposition_actual,
                'Error': typoTransposition_error,
                'ErrorType': ['Typo Transposition' for x in range(len(typoTransposition_error))],
                'SingleError': single_error
            }
        )
        dfs_typo_transposition_error = pd.concat([dfs_typo_transposition_error, df_typo_transposition_error])

    dfs_typo_transposition_error = dfs_typo_transposition_error.drop_duplicates(subset=['Error'])
    return dfs_typo_transposition_error
# ###########################################################


# ###########################################################
def typo_avro_error(df):
    dfs_typo_substituition = pd.DataFrame()
    dict_typoAvroError = {
        'ং': [';'],
        'ঃ': [':', ';'],
        'অ': ['ই', 'প'],
        'আ': ['স'],
        'ই': ['উ', 'অ'],
        'ঈ': ['ই', 'ঊ', 'ও'],
        'উ': ['য়', 'ই'],
        'ঊ': ['উ', 'য়', 'ঈ'],
        'এ': ['ও', 'র'],
        'ও': ['ক', 'এ'],
        'ক': ['ও'],
        'গ': ['ফ', 'হ'],
        'চ': ['ভ'],
        'জ': ['হ', 'ক'],
        'ট': ['ত', 'ড়', 'য়'],
        'ড': ['দ', 'শ', 'ফ'],
        'ণ': ['ন', 'ব', 'ম'],
        'ত': ['র', 'য়'],
        'দ': ['স', 'ফ'],
        'ন': ['ব', 'ম'],
        'প': ['অ'],
        'ফ': ['দ', 'গ'],
        'ব': ['ভ', 'ন'],
        'ভ': ['চ', 'ব'],
        'ম': ['ন'],
        'র': ['এ', 'ত'],
        'ল': ['ক'],
        'শ': ['স', 'আ', 'ড'],
        'স': ['আ', 'দ'],
        'হ': ['গ', 'জ'],
        'ি': ['ে', 'ী'],
        'ী': ['ি'],
        'ু': ['ূ'],
        'ূ': ['ু'],
        'ড়': ['র', 'এ', 'ট'],
        'য়': ['ত', 'উ']
    }

    for i in range(5):
        typoAvro_actual, typoAvro_error, single_error = generate_erroneous_words(df, dict_typoAvroError)
        df_typo_substituition = pd.DataFrame(
            {
                'Word': typoAvro_actual,
                'Error': typoAvro_error,
                'ErrorType': ['Typo (Avro) Substituition' for x in range(len(typoAvro_error))],
                'SingleError': single_error
            }
        )
        dfs_typo_substituition = pd.concat([dfs_typo_substituition, df_typo_substituition])

    dfs_typo_substituition = dfs_typo_substituition.drop_duplicates(subset=['Error'])
    return dfs_typo_substituition
# ###########################################################


# ###########################################################
def typo_bijoy_error(df):
    dict_typoBijoyError = {
        # 'ঁ':[],
        'ং': ['ঙ', 'য়'],
        'ঃ': [';'],
        'অ': ['।', 'ী', 'া'],
        'আ': ['অ'],
        'ও': ['ে', 'ৗ'],
        'ক': ['খ', 'ব', 'ত'],
        'খ': ['ক', 'ভ', 'থ'],
        'গ': ['ঘ', 'হ', 'ড়'],
        'ঘ': ['গ', 'ঞ', 'ঢ়'],
        'ঙ': ['ং', 'য'],
        'চ': ['ছ', 'ট', 'জ'],
        'ছ': ['চ', 'ঠ', 'ঝ'],
        'জ': ['ঝ', 'চ', 'হ'],
        'ঝ': ['জ', 'ছ', 'ঞ'],
        'ঞ': ['হ', 'ঝ', 'ঘ'],
        'ট': ['প', 'চ', 'ঠ'],
        'ঠ': ['ট', 'ফ', 'ছ'],
        'ড': ['ঢ', 'য', 'প'],
        'ঢ': ['ড', 'য়', 'ফ'],
        'ণ': ['ন', 'ল', 'ষ'],
        'ত': ['থ', 'ক', 'দ'],
        'থ': ['ত', 'খ', 'ধ'],
        'দ': ['ধ', 'ত'],
        'ধ': ['দ', 'থ'],
        'ন': ['ণ', 'র', 'স'],
        'প': ['ফ', 'ড', 'ট'],
        'ফ': ['প', 'ঢ', 'ঠ'],
        'ব': ['ভ', 'ক'],
        'ভ': ['ব', 'খ'],
        'ম': ['স', 'শ'],
        'য': ['য়', 'ঙ', 'ড'],
        'র': ['ল', 'ন', 'ে'],
        'ল': ['র', 'ৈ', 'ণ'],
        'শ': ['ম', 'ষ'],
        'ষ': ['স', 'ণ', 'শ'],
        'স': ['ষ', 'ন', 'ম'],
        'হ': ['ঞ', 'জ', 'গ'],
        'া': ['ি', '্'],
        'ি': ['ী', 'ু', 'া'],
        'ী': ['ি', 'ূ'],
        'ু': ['ূ', 'ি'],
        'ূ': ['ু', 'ী'],
        'ৃ': ['ু'],
        'ে': ['ৈ'],
        'ৈ': ['ে', 'ৗ'],
        'ৗ': ['ৈ'],
        'ড়': ['ঢ়', 'গ'],
        'ঢ়': ['ড়', 'ঘ'],
        'য়': ['য', 'ঢ', 'ং']
    }

    dfs_typo_substituition_Bijoy = pd.DataFrame()

    for i in range(5):
        typoBijoy_actual, typoBijoy_error, single_error = generate_erroneous_words(df, dict_typoBijoyError)
        df_typo_substituition_Bijoy = pd.DataFrame(
            {
                'Word': typoBijoy_actual,
                'Error': typoBijoy_error,
                'ErrorType': ['Typo (Bijoy) Substituition' for x in range(len(typoBijoy_error))],
                'SingleError': single_error
            }
        )
        dfs_typo_substituition_Bijoy = pd.concat([dfs_typo_substituition_Bijoy, df_typo_substituition_Bijoy])

    dfs_typo_substituition_Bijoy = dfs_typo_substituition_Bijoy.drop_duplicates(subset=['Error'])
    return dfs_typo_substituition_Bijoy
# ###########################################################


# ###########################################################
def runon_error_gen(word):
    word += random.choice(df.iloc[:, 0])
    return str(word)
# ###########################################################


# ###########################################################
def runon_error(df):
    dfs_runon = pd.DataFrame()
    for i in range(5):
        runon_actual, runon_error = [], []
        single_error = []

        words = df.iloc[:, 0].values
        for i in tqdm(range(len(words))):
            word = words[i]
            rand_word = random.choice(df.iloc[:, 0].values)
            target_word = word + rand_word
            if target_word in df.iloc[:, 0].values:
                continue
            runon_actual.append(word)
            runon_error.append(target_word)

            if (len(runon_actual) == len(runon_error)):
                single_error.append(1)
            else:
                single_error.append(0)

        df_runon = pd.DataFrame({
            'Word': runon_actual,
            'Error': runon_error,
            'ErrorType': ['Run-on Error' for x in range(len(runon_error))],
            'SingleError': single_error
        })
        dfs_runon = pd.concat([dfs_runon, df_runon])

    dfs_runon = dfs_runon.drop_duplicates(subset=['Error'])
    return dfs_runon
# ###########################################################


# ###########################################################
def split_word_random_error(df):
    dfs_splitword_random = pd.DataFrame()
    for i in range(5):
        splitword_actual, splitword_error = [], []

        words = df.iloc[:, 0].values

        for i in tqdm(range(len(words))):
            word = words[i]

            idx = random.randint(1, len(word) - 1)
            splitword_actual.append(word)
            splitword_error.append(word[:idx] + ' ' + word.split(word[:idx])[-1])

        df_splitword_random = pd.DataFrame({
            'Word': splitword_actual,
            'Error': splitword_error,
            'ErrorType': ['Split-word Error (Random)' for x in range(len(splitword_error))],
        })
        dfs_splitword_random = pd.concat([dfs_splitword_random, df_splitword_random])

    dfs_splitword_random = dfs_splitword_random.drop_duplicates(subset=['Error'])
    return dfs_splitword_random
# ###########################################################


# ###########################################################
def split_word_error_left_both(df):
    splitword_actual_left, splitword_error_leftonly = [], []
    splitword_actual_both, splitword_error_both = [], []

    for idx, word in enumerate(df.iloc[:, 0].values):
        if idx % 10000 == 0:
            print(f'Checked - {idx}\t Found - (left only): {len(splitword_error_leftonly)}\t and (both): {len(splitword_error_both)}')
        similar_words_left = df.loc[df['word'].str.startswith(word[:2])].iloc[:, 0].values
        if len(similar_words_left) == 0:
            continue

        for i in range(len(word)):
            word_left = word[:i]
            if word_left in similar_words_left:
                word_right = word.split(word_left)[-1]
                splitword_actual_left.append(word)
                splitword_error_leftonly.append(word_left + ' ' + word_right)
                similar_words_right = df.loc[df['word'].str.startswith(word_right[:1])].iloc[:, 0].values
                if word_right in similar_words_right:
                    splitword_actual_both.append(word)
                    splitword_error_both.append(word_left + ' ' + word_right)
                break

        df_splitword_left = pd.DataFrame({
            'Word': splitword_actual_left,
            'Error': splitword_error_leftonly,
            'ErrorType': ['Split-word Error (Left)' for x in range(len(splitword_actual_left))],
        })

        df_splitword_both = pd.DataFrame({
            'Word': splitword_actual_both,
            'Error': splitword_error_both,
            'ErrorType': ['Split-word Error (both)' for x in range(len(splitword_error_both))],
        })

    return df_splitword_left, df_splitword_both
# ###########################################################


# ###########################################################
def split_word_error_right(df):
    splitword_actual_right, splitword_error_rightonly = [], []
    for idx, word in enumerate(df.iloc[:, 0].values):
        if idx % 10000 == 0:
            print(f"Cheked {idx} words and found {count} words")

        for i in range(len(word) - 1, 0, -1):
            temp_word = word[i:]
            similar_words_right = df.loc[df['word'].str.startswith(temp_word)].iloc[:, 0].values
            if temp_word in similar_words_right:
                splitword_actual_right.append(word)
                splitword_error_rightonly.append(temp_word)

    df_splitword_right = pd.DataFrame({
        'Word': splitword_actual_right,
        'Error': splitword_error_rightonly,
        'ErrorType': ['Split-word Error (Right)' for x in range(len(splitword_actual_right))],
    })

    return df_splitword_right
# ###########################################################


# ###########################################################
if __name__ == '__main__':
    df = pd.read_csv('./dfs/df_all_words.csv')
    df.drop(
        [
            df[df['word'] == 'অশুভকককততকককবডহঅশুভকর'].index[0],
            df[df['word'] == 'অশুভকককততকককবডহঅশুভঙ্কর'].index[0],
        ],
        inplace=True
    )

    df_cognitive_phonetic = cognitive_phonetic_error(df)
    df_visual = visual_error(df)
    df_combinedchar_visual_error = visual_error_combined_char(df)
    df_typo_insertion_error = typo_insertion_error(df)
    df_typo_deletion_error = typo_deletion_error(df)
    df_typo_transposition_error = typo_insertion_error(df)
    df_typo_substituition_avro = typo_avro_error(df)
    df_typo_substituition_bijoy = typo_bijoy_error(df)
    df_runon = runon_error(df)
    df_splitword_random = split_word_random_error(df)
    df_splitword_left, df_splitword_both = split_word_error_left_both(df)
    df_splitword_right = split_word_error_right(df)
    #
    df_homonym_error = pd.read_excel('./dfs/homonymerrors.xlsx')
    df_homonym_error['ErrorType'] = ['Homonym Error' for x in range(len(df_homonym_error))]

    final_df = pd.concat([
        df_cognitive_phonetic,
        df_visual,
        df_combinedchar_visual_error,
        df_typo_insertion_error,
        df_typo_deletion_error,
        df_typo_transposition_error,
        df_typo_substituition_avro,
        df_typo_substituition_bijoy,
        df_runon,
        df_splitword_random,
        df_splitword_left,
        df_splitword_both,
        df_splitword_right,
        df_homonym_error
    ])

    final_df.to_csv(
        './dfs/sec_dataset_III.csv', index=False
    )