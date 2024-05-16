import os, time
import pandas as pd
import platform

from tqdm import trange
from typing import Tuple
from random import shuffle
from collections import defaultdict

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


def randomize(size: int, df: pd.DataFrame) -> pd.DataFrame:
    lst: list[int] = list(range(size))
    shuffle(lst)
    return df.loc[lst]



def combine_datasets(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    method: str | None = None,
) -> pd.DataFrame:
    """
        Output the combined dataset with the specified method, default is None

        Parameters:
            df1 (pd.DataFrame): The first DataFrame
            df2 (pd.DataFrame): The second DataFrame
            method (str): The method to combine the datasets
                - None, Other: Concatenate the datasets
                - Random: Randomly combine the datasets
        
        Returns:
            pd.DataFrame: The combined DataFrame
    """
    # Concatenate the datasets
    # question, answer, label, focus_area
    #
    # df1: HealthCare_NLP
    #     - question	answer	source	focus_area	label
    # df2: Comprehensive_QA
    #     - qtype	Diseases(full name)	Question	Answer
    # df: Combined
    #     - question	answer	label	focus_area
    dct: defaultdict[str, list] = defaultdict(list)
    s1, s2 = df1.shape[0], df2.shape[0]
    print('Extracting HealthCare_NLP ...')
    for i in trange(s1):
        dct['question'].append(df1.loc[i, 'question'])
        dct['answer'].append(df1.loc[i, 'answer'])
        dct['label'].append(df1.loc[i, 'label'])
        dct['focus_area'].append(df1.loc[i, 'focus_area'])
    print('Finished HealthCare_NLP...'), stop()
    print('Extracting Comprehensive_QA ...')
    for i in trange(s2):
        dct['question'].append(df2.loc[i, 'Question'])
        dct['answer'].append(df2.loc[i, 'Answer'])
        dct['label'].append(df2.loc[i, 'qtype'])
        dct['focus_area'].append(df2.loc[i, 'Diseases(full name)'])
    print('Finished Comprehensive_QA...')

    tot = s1 + s2
    print('Combining ...')
    df = pd.DataFrame(dct)
    if method == 'Random':
        df = randomize(tot, df)
    print('Finished combining...'), stop()
    return df


def k_fold(
    combined_df: pd.DataFrame, 
    k: int = 10,
    method: str | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if k < 10:
        raise ValueError('k must be greater than 10')
    
    size = combined_df.shape[0]
    temp_fold_train: list[defaultdict[str, list[str]]] = []
    temp_fold_test: list[defaultdict[str, list[str]]] = []
    for _ in range(k):
        temp_fold_train.append(defaultdict(list))
        temp_fold_test.append(defaultdict(list))

    if method == 'Random':
        # Already randomized
        number = size // k + 1
        for i in trange(size):
            question = combined_df.loc[i, 'question']
            answer = combined_df.loc[i, 'answer']
            label = combined_df.loc[i, 'label']
            focus_area = combined_df.loc[i, 'focus_area']
            for j in range(k):
                if i // number == j:
                    temp_fold_test[j]['question'].append(question)
                    temp_fold_test[j]['answer'].append(answer)
                    temp_fold_test[j]['label'].append(label)
                    temp_fold_test[j]['focus_area'].append(focus_area)
                else:
                    temp_fold_train[j]['question'].append(question)
                    temp_fold_train[j]['answer'].append(answer)
                    temp_fold_train[j]['label'].append(label)
                    temp_fold_train[j]['focus_area'].append(focus_area)
    else:
        for i in trange(size):
            question = combined_df.loc[i, 'question']
            answer = combined_df.loc[i, 'answer']
            label = combined_df.loc[i, 'label']
            focus_area = combined_df.loc[i, 'focus_area']
            for j in range(k):
                if i % k == j:
                    temp_fold_test[j]['question'].append(question)
                    temp_fold_test[j]['answer'].append(answer)
                    temp_fold_test[j]['label'].append(label)
                    temp_fold_test[j]['focus_area'].append(focus_area)
                else:
                    temp_fold_train[j]['question'].append(question)
                    temp_fold_train[j]['answer'].append(answer)
                    temp_fold_train[j]['label'].append(label)
                    temp_fold_train[j]['focus_area'].append(focus_area)
        
    for i in range(k):
        train = pd.DataFrame(temp_fold_train[i])
        test = pd.DataFrame(temp_fold_test[i])
        tocsv(train, f'./process/k_fold/train_{i}{f"_{method}" if method == 'Random' else ""}')
        tocsv(test, f'./process/k_fold/test_{i}{f"_{method}" if method == 'Random' else ""}')
        print(f'Data {i} saved... ({method if method == 'Random' else "General"})'), stop()


def tocsv(df: pd.DataFrame, filename: str) -> None:
    """
        Write a DataFrame to a CSV file

        Parameters:
            df (pd.DataFrame): The DataFrame to write
    """
    df.to_csv(f'{filename}.csv', index=False)


if __name__ == '__main__':
    # HealthCare_NLP
    df_HealthCare_NLP = pd.read_csv('data/HealthCare_NLP_v1.csv')
    df_HealthCare_NLP = check_labels(df_HealthCare_NLP, 'HealthCare_NLP')
    # Comprehensive_QA
    df_Comprehensive_QA = pd.read_csv('data/Comprehensive_QA_v1.csv')
    df_Comprehensive_QA = check_labels(df_Comprehensive_QA, 'Comprehensive_QA')
    clear(), print('Two Datasets checked...'), stop()

    # If the datasets are checked, combine them
    print('Combining datasets...')
    df_combined = combine_datasets(df_HealthCare_NLP, df_Comprehensive_QA)
    tocsv(df_combined, './process/combined/combined'), clear()
    print('Combined dataset saved...'), stop()
    print('Splitting datasets...')
    k_fold(df_combined, 10), stop()
    ############################################################################################################
    df_combined = combine_datasets(df_HealthCare_NLP, df_Comprehensive_QA, 'Random')
    tocsv(df_combined, './process/combined/combined_random'), clear()
    print('Randomly combined dataset saved...'), stop()
    print('Splitting datasets...')
    k_fold(df_combined, 10, 'Random')
    ############################################################################################################
    print('All done!'), stop(2), clear()
