#===============================================================================
#     Modified Functions for Functions
#===============================================================================
import pandas as pd
import numpy as np
import random
import os
import sys
import logging
ROOT_DIR = os.path.abspath("./config")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import data_config as Dconfig
import argparse

logging.basicConfig(level=logging.INFO)

#Functions

def otherCap(x):
    for letter in x:
        if letter.isupper():
            return 1
    return 0

def propVow(x):
    vowels = 'aeiouAEIOU'
    numVow = 0
    for letter in x:
        if letter in vowels:
            numVow += 1
    return numVow / len(x)

def frontWord(x, array, df):
    if x > 0:
        return array.index(df['Word'][x-1])
    else:
        return
    
def backWord(x, array, df):
    if x < len(df.index) - 1:
        return array.index(df['Word'][x+1])
    else:
        return


def feature_gen_4_pred(filename = Dconfig.PRED_DATASET_PATH):
    
    logging.info('Feature Generation has begun')
    df = pd.read_csv(filename, sep='\t', encoding='unicode_escape')
    
    df['isFirstCap'] = df['Word'].apply(lambda x: 1 if x[0].isupper() else 0)

    df['Length'] = df['Word'].apply(lambda x: len(x))

    df['endY'] = df['Word'].apply(lambda x: 1 if x[-1] == 'y' else 0)

    df['isNNP'] = df['POS'].apply(lambda x: 1 if x == 'NNP' else 0)

    df['isJJ'] = df['POS'].apply(lambda x: 1 if x == 'JJ' else 0)

    df['isCD'] = df['POS'].apply(lambda x: 1 if x == 'CD' else 0)

    df['otherCap'] = df['Word'].apply(lambda x: otherCap(x))

    df['endan'] = df['Word'].apply(lambda x: 1 if x[-2:len(x)] == 'an' else 0)

    df['isNum'] = df['Word'].apply(lambda x: 1 if x.isnumeric() else 0)

    df['endS'] = df['Word'].apply(lambda x: 1 if x[-1] == 's' else 0)

    df['endish'] = df['Word'].apply(lambda x: 1 if x[-3:len(x)] == 'ish' else 0)

    df['endese'] = df['Word'].apply(lambda x: 1 if x[-3:len(x)] == 'ese' else 0)

    df['propVow'] = df['Word'].apply(lambda x: propVow(x))

    logging.info('Simple features have been generated, moving on to difficult features')
    word_array = df.Word.unique().tolist()
    df['frontWord'] = df['Unnamed: 0'].apply(lambda x: frontWord(x, word_array, df))

    df['backWord'] = df['Unnamed: 0'].apply(lambda x: backWord(x, word_array, df))

    logging.info('All features done...')

    return df




