import os, time
import pandas as pd
import platform

from tqdm import trange
from typing import Tuple, Dict

qlabels = [
    'symptoms', 'treatment', 'information', 'causes',
    'tests', 'prevention', 'susceptibility', 'research'
]
s = 0.5

stop = lambda sec = s: time.sleep(sec)
"""
    Parameter: 
        sec (int): The number of seconds to sleep
"""
clear = lambda: os.system('cls') if platform.system() == 'Windows' else os.system('clear')
"""
    Clear terminal based on the OS
"""

def check_labels(df: pd.DataFrame, whichdataset: str) -> pd.DataFrame:
    """
        Check if there are any missing labels in the dataset

        Parameters:
            df (pd.DataFrame): The dataset to check
            whichdataset (str): The dataset name
    """
    if whichdataset == 'Comprehensive_QA':
        print('Checking Comprehensive_QA dataset...'), stop()
        # symptoms
        df.loc[df['qtype'] == 'complications', 'qtype'] = 'symptoms'
        df.loc[df['qtype'] == 'stages', 'qtype'] = 'symptoms'
        print('\tsymptoms checked...'), stop()
        # information
        df.loc[df['qtype'] == 'frequency', 'qtype'] = 'information'
        df.loc[df['qtype'] == 'inheritance', 'qtype'] = 'information'
        df.loc[df['qtype'] == 'outlook', 'qtype'] = 'information'
        df.loc[df['qtype'] == 'support groups', 'qtype'] = 'information'
        df.loc[df['qtype'] == 'considerations', 'qtype'] = 'information'
        df.loc[df['qtype'] == 'genetic changes', 'qtype'] = 'information'
        print('\tinformation checked...'), stop()
        # tests
        df.loc[df['qtype'] == 'exams and tests', 'qtype'] = 'tests'   
        print('\ttests checked...'), stop()
        print('Comprehensive_QA dataset checked...'), stop(), clear()
    elif whichdataset == 'HealthCare_NLP':
        print('Checking HealthCare_NLP dataset...'), stop()
        labels = df['label'].unique()
        for label in labels:
            if label not in qlabels:
                raise ValueError(f'Invalid label: {label}')
            print(f"\t{label} checked..."), stop()
        if len(labels) != len(qlabels):
            raise ValueError('Some labels are missing')
        print("Done!"), stop(), clear()
        print('HealthCare_NLP dataset checked...'), stop()
    else:
        raise ValueError('Invalid dataset name')
    return df

def combine_datasets_with_same_label(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    k: int = 10
) -> Tuple[Tuple[pd.DataFrame], Tuple[pd.DataFrame]]:
    """
        Combine two datasets with the same label

        Parameters:
            df1 (pd.DataFrame): The first dataset
            df2 (pd.DataFrame): The second dataset
    """
    k_fold_train: list[pd.DataFrame] = []
    k_fold_test: list[pd.DataFrame] = []
    if k < 10:
        raise ValueError('k must be greater than 10')
    


    s1, s2 = len(df1), len(df2) # the size of the datasets
    temp_fold_train: list[Dict[str, list[str]]] = []
    temp_fold_test: list[Dict[str, list[str]]] = []
    for _ in range(k):
        temp_fold_train.append({
            'question': [],
            'answer': [],
            'label': []
        })
        temp_fold_test.append({
            'question': [],
            'answer': [],
            'label': []
        })
    for i in trange(s1):
        question = df1.iloc[i]['question']
        answer = df1.iloc[i]['answer']
        label = df1.iloc[i]['label']
        for j in range(k):
            if i % k == j:
                temp_fold_test[j]['question'].append(question)
                temp_fold_test[j]['answer'].append(answer)
                temp_fold_test[j]['label'].append(label)
            else:
                temp_fold_train[j]['question'].append(question)
                temp_fold_train[j]['answer'].append(answer)
                temp_fold_train[j]['label'].append(label)
    
    for i in trange(s2):
        question = df2.iloc[i]['Question']
        answer = df2.iloc[i]['Answer']
        label = df2.iloc[i]['qtype']
        for j in range(k):
            if i % k == j:
                temp_fold_test[j]['question'].append(question)
                temp_fold_test[j]['answer'].append(answer)
                temp_fold_test[j]['label'].append(label)
            else:
                temp_fold_train[j]['question'].append(question)
                temp_fold_train[j]['answer'].append(answer)
                temp_fold_train[j]['label'].append(label)

    for i in range(k):
        k_fold_train.append(pd.DataFrame(temp_fold_train[i]))
        k_fold_test.append(pd.DataFrame(temp_fold_test[i]))

    return tuple(k_fold_train), tuple(k_fold_test)
    

def tocsv(df: pd.DataFrame, filename: str) -> None:
    """
        Write a DataFrame to a CSV file

        Parameters:
            df (pd.DataFrame): The DataFrame to write
    """
    df.to_csv(f'{filename}.csv', index=False)

if __name__ == '__main__':
    # HealthCare_NLP
    df_HealthCare_NLP = pd.read_csv('data/HealthCare_NLP.csv')
    df_HealthCare_NLP = check_labels(df_HealthCare_NLP, 'HealthCare_NLP')
    # Comprehensive_QA
    df_Comprehensive_QA = pd.read_csv('data/Comprehensive_QA.csv')
    df_Comprehensive_QA = check_labels(df_Comprehensive_QA, 'Comprehensive_QA')
    clear(), print('Two Datasets checked...'), stop()

    # If the datasets are checked, combine them
    print('Combining datasets...')
    train, test = combine_datasets_with_same_label(df_HealthCare_NLP, df_Comprehensive_QA)
    tocsv(train[0], 'train'), tocsv(test[0], 'test')

# question answer label
# random 
# 32000 
# 20000 - 30000
# 2000

"""
'symptoms'
    - 'complications'
    - 'stages'
'treatment'
'information'
    - 'frequency'
    - 'inheritance'
    - 'outlook'
    - 'support groups'
    - 'considerations'
    - 'genetic changes'
'causes'
'tests':
    - 'exams and tests'
'prevention'
'susceptibility'
'research'
"""
