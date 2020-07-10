#===============================================================================
#     Data Transformation and Splitting
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

#Functions for Feature Generation

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

def Tag2Num(x, array):
    return array.index(x)


def feature_gen(filename = Dconfig.DATASET_PATH):
    '''
    Generates various features for a dataframe to use for modelling
    the new data will now be sent to a file specified in configs
    +Inputs:
        filename: the path of file that holds the data
    '''

    #Reads the file (a csv but can be txt) and sets it to a Pandas dataframe
    logging.info('Feature Generation has begun')
    df = pd.read_csv(filename, sep='\t', encoding='unicode_escape')

    #Generates new columns of the dataframe based on various features
    #(capitalization, if one is a part of speech, etc.). These are stored as numbers
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
    tag_array = df.Tag.unique().tolist()
    df['TagNum'] = df['Tag'].apply(lambda x: Tag2Num(x, tag_array))
    word_array = df.Word.unique().tolist()
    df['frontWord'] = df['Unnamed: 0'].apply(lambda x: frontWord(x, word_array, df))

    df['backWord'] = df['Unnamed: 0'].apply(lambda x: backWord(x, word_array, df))

    logging.info('All features done... saving to file')

    #Saves the new dataframe as a csv file    
    df.to_csv(Dconfig.FEATURES_DATASET_PATH, encoding = 'unicode-escape')


def data_split(filename = Dconfig.FEATURES_DATASET_PATH, mode = "BOTH"):
    '''
    Splits the data into training and testing batches, and saves them
    +Inputs:
        filename: the name of the filepath for the featured dataset
        mode: can either split the data into training and testing, or
            set all to testing, or set all to training
    '''

    #Reads the featured csv file and writes it in as a Pandas dataframe
    logging.info('Data Splitting has begun.')
    df = pd.read_csv(filename, encoding='unicode_escape')

    #If the mode is set to both training and testing
    if mode == "BOTH":

        #Splits the words by sentence, and then randomly takes 25% 
        #of them and puts them into the testing set
        sentences_group = df.groupby(['Sentence #'])
        test_sentences = []
        test_dfs = []
        train_dfs = []
        max_sent = int(df['Sentence #'].max())
        for i in range(round(max_sent / 4)):
            found = False
            while found == False:
                num = random.randint(1, max_sent)
                if not num in test_sentences:
                    test_sentences.append(num)
                    test_dfs.append(sentences_group.get_group(num))
                    found = True

        #takes the testing sentences, and removes them from the training set        
        test_df = pd.concat(test_dfs)
        drop_list = test_df['Unnamed: 0'].tolist()
        train_df = df.copy().drop(drop_list)

        #takes the data and labels of the training and testing sets, and puts them in txt files
        data_train = train_df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
        label_train = train_df['TagNum'].values
        data_test = test_df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
        label_test = test_df['TagNum'].values

        np.savetxt(Dconfig.DATA_TRAIN_PATH, data_train)
        np.savetxt(Dconfig.LABEL_TRAIN_PATH, label_train)
        np.savetxt(Dconfig.DATA_TEST_PATH, data_test)
        np.savetxt(Dconfig.LABEL_TEST_PATH, label_test)

        logging.info('Split complete')

    #If the mode is set to evaluation (all testing)
    elif mode == "EVAL":

        #Takes all of the data and splits the data from label columns and saves them in txt files
        #for testing
        data_test = df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
        label_test = df['TagNum'].values
        np.savetxt(Dconfig.DATA_TEST_PATH, data_test)
        np.savetxt(Dconfig.LABEL_TEST_PATH, label_test)

        logging.info('Full evaluation split complete')

    #If the mode is set to all training
    elif mode == "TRAIN":

        #Takes all of the data and splits the data from label columns and saves them in txt files
        #for training
        data_train = df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
        label_train = df['TagNum'].values
        np.savetxt(Dconfig.DATA_TRAIN_PATH, data_train)
        np.savetxt(Dconfig.LABEL_TRAIN_PATH, label_train)

        logging.info('Full train split complete')

    #If the mode is set to a nonvalid mode
    else:
        logging.debug('Please put in a valid mode, or none to do a normal split')


#argparse code to allow command line functionality
parser = argparse.ArgumentParser(description='Methods for feature generation and splitting of data')
parser.add_argument("command", metavar="<command>", help="'feature_gen' or 'split' or 'split_eval' or 'split_train'",)
args = parser.parse_args()
assert args.command in ['feature_gen', 'split', 'split_eval', 'split_train'], "invalid parsing 'command'"

if args.command == "feature_gen":
        feature_gen()      
elif args.command == "split":
        data_split()
elif args.command == "split_eval":
    data_split(mode = 'EVAL')
else:
    data_split(mode = 'TRAIN')





