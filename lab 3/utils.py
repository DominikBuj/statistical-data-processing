import os, json
from collections import Counter, defaultdict
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

heatmap_colors = cm.get_cmap("Reds", 1024).with_extremes(under="white")

def process_data(results_path: str) -> Tuple[List, List]:
    '''Returns the dataset tuple where the first element 
    contains a list with input samples and the second contains a list with labels.

    Parameters
    ----------
    results_path : str
        Path to json files.


    Example
    ----------
    X, Y = process_data('../results')

    '''
    
    results_jsons = []
    for el in os.listdir(results_path):
        filepath = f'{results_path}/{el}'
        if not os.path.isfile(filepath) or len(filepath) < 4 or filepath[-5:] != '.json':
            print(el, '- not a correct file, skipping')
            continue
        with open(filepath, 'r') as fp:
            results_jsons += [{**json.load(fp), 'filename': el}]
            
            
    rows = []
    for user in results_jsons:
    #     if user['filename'] in ['sg.json', 'af.json']: continue
        user_props = {'filename': user['filename']}
        for user_prop in ['rating', 'speed', 'typos', 'age', 'sex', 'hand', 'phone']:
            user_props['user_'+user_prop] = user[user_prop]
        for test_idx, test in enumerate(user['tests']):
            test_props = {'test_id': f'{user["filename"]}_{test_idx}'}
            for test_prop in ['mode', 'startTime', 'endTime']:
                test_props['test_'+test_prop] = test[test_prop]
            for sentence in test['sentences']:
                sentence_props = {}
                for sentence_prop in ['idx', 'sentence', 'startTime', 'endTime']:
                    sentence_props['sentence_'+sentence_prop] = sentence[sentence_prop]
                previous_wrong = False
                for key in sentence['pressedKeys']:
                    rows += [{
                        **user_props,
                        **test_props,
                        **sentence_props,
                        'key_position': key['position'],
                        'key_is_correct': key['correct'],
                        'key_is_wrong': not key['correct'],
                        'key_is_wrong_following': previous_wrong,
                        'key': key['key']['key'],
                        'label': key['key']['label'],
                        'original_key': key['key']['original_key'],
                        'original_label': key['key']['original_label'],
                        'expected_key': key['expected_key'].lower() if 'expected_key' in key else 'NONE',
                        'key_time': key['time'],
                    }]
                    previous_wrong = not key['correct']
                    
                

    results_df = pd.DataFrame(rows)
    results_df['test_date'] = pd.to_datetime(results_df['test_startTime'], unit='ms', yearfirst=True)
    results_df['test_time'] = results_df['test_endTime'] - results_df['test_startTime']

    key_number = results_df.groupby('test_id')['key'].count().to_dict()
    results_df['test_number_of_taps'] = results_df['test_id'].apply(lambda x: key_number[x])
    results_df['test_taps_per_second'] = (results_df['test_number_of_taps'] / results_df['test_time'])*1000
    number_of_wrong = results_df.groupby('test_id')['key_is_wrong'].sum().to_dict()
    results_df['number_of_wrong']  = results_df['test_id'].apply(lambda x: number_of_wrong[x])
    results_df['sentence_time']  = results_df['sentence_endTime'] - results_df['sentence_startTime']
    results_df['sentence_time_per_character']  = results_df['sentence_time'] / results_df['sentence_sentence'].str.len()
    results_df['key_is_wrong_first'] = (results_df['key_is_correct'] == False)&(results_df['key_is_wrong_following'] == False)
    number_of_wrong_first = results_df.groupby('test_id')['key_is_wrong_first'].sum().to_dict()
    results_df['number_of_wrong_first']  = results_df['test_id'].apply(lambda x: number_of_wrong_first[x])

    users_results = results_df.groupby('test_id').head(1).groupby(['filename', 'test_mode', 'user_sex', 'user_age', 'user_hand', 'user_phone', 'user_rating', 'user_speed', 'user_typos'])[['test_time', 
        'number_of_wrong_first', 'number_of_wrong', 'test_number_of_taps', 'test_taps_per_second']].mean().copy().reset_index()
    users_results['test_time'] = users_results['test_time']/1000


    users_results['user_age']    = pd.to_numeric(users_results['user_age'])  
    users_results['user_rating'] = pd.to_numeric(users_results['user_rating']) 
    users_results['user_speed']  = pd.to_numeric(users_results['user_speed']) 
    users_results['user_typos']  = pd.to_numeric(users_results['user_typos'])

    all_letters_in_text = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    all_letters_entered = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    symbol_map = {
        'backspace': 'BACKS',
        '\n'       : 'ENTER',
        ' '        : 'SPACE',
        'shift'    : 'SHIFT'
    }
    all_letters_entered_v = list(map(lambda x: symbol_map[x] if x in symbol_map else x, all_letters_entered))
    all_letters_in_text_v = list(map(lambda x: symbol_map[x] if x in symbol_map else x, all_letters_in_text))
    
    text_character_count = dict(Counter(''.join(results_df['sentence_sentence'].str.lower().unique())))
    users                = results_df['filename'].unique()

    def get_typo_matrix(test_mode=None, normalize=True):
        q = []
        q_key_wrong = '(key_is_wrong_first == True)'
        if test_mode is not None:
            q_test_mode = f'(test_mode == "{test_mode}")'
            q.append(q_test_mode)
        
        l = []
        for user in users:
            m = []
            q_filename = f'(filename == "{user}")'
            number_of_tests = results_df.query(' & '.join(q + [q_filename])).copy().reset_index().groupby('test_id').head(1).count()['filename']
    #         print(number_of_tests, user)
            conf_matrix = (
                results_df
                .query(' & '.join(q + [q_filename, q_key_wrong]))
                .groupby(['key', 'expected_key']).count()['filename'].reset_index().rename(columns={'filename': 'count'})
                .pivot(index='key', columns='expected_key', values='count')
                .fillna(0)
                /number_of_tests
            ).to_dict()
            for letter_expected in all_letters_in_text:
                row = []
                for letter_entered in all_letters_entered:
                    # pobierz liczbe literowek dla danej litery i dodaj do slownika
                    val = 0
                    if letter_expected in conf_matrix:
                        if letter_entered in conf_matrix[letter_expected]:
                            val = conf_matrix[letter_expected][letter_entered]
                            if normalize:
                                val = (1000*val)/text_character_count[letter_expected]
                    row += [val]
                m += [row]
            l += [np.array(m)]
        return np.array(l)


    typos_per_letter = {'letter': all_letters_entered_v}

    typo_matrix = defaultdict(list)

    for test_mode in ['normal', 'keydrop']:
        l = get_typo_matrix(test_mode, False)

        s = l[0]
        typo_matrix[test_mode] += [l[0]]
    #     typo_matrix[test_mode] += [l[0].sum(axis=1)]
        for i in range(1, len(l)):
            s += l[i]
            typo_matrix[test_mode] += [l[i]]
    #         typo_matrix[test_mode] += [l[i].sum(axis=1)]

        s = (100 * s) / float(len(users))
        s = s.astype('int')

        typos_per_letter[test_mode] = list(s.sum(axis=1))

        # qwe = pd.DataFrame(s, columns=all_letters_entered_v, index=all_letters_in_text_v)
        # print(f'# of typos in {test_mode}:', qwe.sum().sum())
        
    # zxc = pd.DataFrame(typos_per_letter)
    # zxc['kdrp/norm'] = zxc.apply(lambda x: round(x['keydrop']/x['normal'], 2) if x['normal'] >0 else 0, axis=1)
    # zxc['norm/kdrp'] = zxc.apply(lambda x: round(x['normal']/x['keydrop'], 2) if x['keydrop'] >0 else 0, axis=1)
    # print(zxc.to_markdown(index=None))

    # typo_matrix_normal = typo_matrix['normal']
    # typo_matrix_normal_meta = [results_df[results_df['filename'] == user]['user_sex'].head(1).values[0] for user in users]
    # typo_matrix_keydrop = typo_matrix['keydrop']
    # typo_matrix_keydrop_meta = [results_df[results_df['filename'] == user]['user_sex'].head(1).values[0] for user in users]
    # X = typo_matrix_normal      + typo_matrix_keydrop
    # Y = typo_matrix_normal_meta + typo_matrix_keydrop_meta

    typo_matrix_normal = typo_matrix['normal']
    typo_matrix_normal_meta = ['normal']*len(typo_matrix_normal)
    typo_matrix_keydrop = typo_matrix['keydrop']
    typo_matrix_keydrop_meta = ['keydrop']*len(typo_matrix_keydrop)
    X = typo_matrix_normal      + typo_matrix_keydrop
    Y = typo_matrix_normal_meta + typo_matrix_keydrop_meta

    return (X, Y)

def get_characters():
    return ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def show_matrix(m:Union[List[List], np.ndarray], title:str='') -> None:
    '''Shows the typo matrix 

    Parameters
    ----------
    m : List[List]
        Typo matrix
    title: str
        Chart title

    Example
    ----------
    show_matrix(X[0])

    '''
    fig, ax = plt.subplots(figsize=(12,10))
    plt.title(title)
    
    res = sns.heatmap(m, annot=True, xticklabels=True, yticklabels=True, fmt='g', ax=ax, cmap=heatmap_colors, linewidths=1, linecolor='grey')
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
    characters = get_characters()
    plt.yticks(rotation=0, ticks=range(len(characters)), labels=characters, va='top')
    plt.xticks(rotation=45,ticks=range(len(characters)), labels=characters, ha='left')
    plt.xlabel('Key clicked')
    plt.ylabel('Key expected')
    plt.show()