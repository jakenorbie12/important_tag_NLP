#===============================================================================
#     Data Transformation and Splitting
#===============================================================================
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import random
import nltk
import os
import sys
import logging
ROOT_DIR = os.path.abspath("./config")
#print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import data_config as Dconfig
import argparse

#For First Time Only
#nltk.download('punkt')

logging.basicConfig(level=logging.INFO)

#Functions for Feature Generation

def isFirstCap(x):
    if x[0].isupper():
        return 1
    else:
        return 0

def Length(x):
    return len(x)

def endY(x):
    if x[-1] == 'y':
        return 1
    else:
        return 0

def endan(x):
    if x[-2:len(x)] == 'an':
        return 1
    else:
        return 0

def isNum(x):
    if x.isnumeric():
        return 1
    else:
        return 0

def endS(x):
    if x[-1] == 's':
        return 1
    else:
        return 0
    
def endish(x):
    if x[-3:len(x)] == 'ish':
        return 1
    else:
        return 0

def endese(x):
    if x[-3:len(x)] == 'ese':
        return 1
    else:
        return 0

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

def Array2Num(x, array):
    return array.index(x)


def feature_gen(filename = Dconfig.DATASET_PATH, mode = 'TRAIN', data = None):
    '''
    Generates various features for a dataframe to use for modelling
    the new data will now be sent to a file specified in configs
    +Inputs:
        filename: the path of file that holds the data
        mode: determines if this is for predicting or training (not predicting) or evaluation
    '''

    #Reads the file (a csv but can be txt) and sets it to a Pandas dataframe
    logging.info('Feature Generation has begun')

    if mode == 'TRAIN' or mode == 'EVAL':
        df = pd.read_csv(filename, sep='\t', encoding='unicode_escape')

    #Takes the parameter dataframe for prediction
    elif mode == 'PREDICT':
        df = data

    #Generates new columns of the dataframe based on various features
    #(capitalization, if one is a part of speech, etc.). These are stored as numbers
    function_dict = {'isFirstCap': isFirstCap, 'Length': Length, 'endY': endY, 'otherCap': otherCap, 'endan': endan,
                  'isNum': isNum, 'endS': endS, 'endish': endish, 'endese': endese, 'propVow': propVow}
    for f in function_dict:
        df[f] = df['Word'].apply(lambda x: function_dict[f](x))

    logging.info('Simple features have been generated, moving on to difficult features')

    #If the mode is training then the part of speech is converted to a number and a list of the order is stored
    if mode == 'TRAIN':
        #POS_array gives a list of the POS in order of coming up
        POS_array = df.POS.unique().tolist()
        df['POSNum'] = df['POS'].apply(lambda x: Array2Num(x, POS_array))
        POS_file = open('./data/process/POS_array.txt', 'w')
        POS_file.write(str(POS_array))
        POS_file.close()

        word_data = df['Word'].values
        word_vec = [nltk.word_tokenize(title) for title in word_data]
        model = Word2Vec(word_vec, size=24, window=5, min_count=0, workers=4)
        model.save('./data/process/word2vec.model')
        wv = KeyedVectors.load('./data/process/word2vec.model')
        for i in range(24):
            df['WordVector' + str(i)] = df['Word'].apply(lambda x: wv[x][i] if x in wv else None)

    #If the mode is prediction or evaluation then the stored list is loaded and converts each POS to a number
    elif mode == 'PREDICT' or mode == 'EVAL':
        POS_numfile = open('./data/process/POS_array.txt', 'r')
        POS_array = POS_numfile.read()
        POS_array = POS_array.strip("[]")
        POS_array = POS_array.split(',')
        for idx, word in enumerate(POS_array):
            if word == " '":
                POS_array[idx] = ','
            else:
                new_word = word.strip()
                new_word = new_word.strip("''")
                POS_array[idx] = new_word
        POS_array.pop(POS_array.index(',') + 1)
        df['POSNum'] = df['POS'].apply(lambda x: POS_array.index(x))
    

    word_array = df.Word.unique().tolist()
    df['frontWord'] = df['Unnamed: 0'].apply(lambda x: frontWord(x, word_array, df))

    df['backWord'] = df['Unnamed: 0'].apply(lambda x: backWord(x, word_array, df))

    #If the mode is for training and not predicting...
    # Generates tag num from tag name,
    #Saves the tag array so it will be used later to change the predicted tags nums to tags
    if mode == 'TRAIN':
        tag_array = df.Tag.unique().tolist()
        tag_file = open('./data/process/tag_array.txt', 'w')
        tag_file.write(str(tag_array))
        tag_file.close()
        df['TagNum'] = df['Tag'].apply(lambda x: Array2Num(x, tag_array))
    
        
        logging.info('All features done... saving to file')

        #Saves the new dataframe as a csv file  
        df.to_csv(Dconfig.FEATURES_DATASET_PATH, encoding = 'unicode-escape')

    #If the mode is full evaluation, then TagNum is made from the tag_array file
    elif mode == 'EVAL':
        tag_file = open('./data/process/tag_array.txt', 'r')
        tag_array = tag_file.read()
        tag_array = tag_array.strip("[]")
        tag_array = tag_array.split(',')
        for idx, word in enumerate(tag_array):
                new_word = word.strip()
                new_word = new_word.strip("''")
                tag_array[idx] = new_word
        df['TagNum'] = df['Tag'].apply(lambda x: tag_array.index(x))

    #else if the mode is for prediction, returns the df
    elif mode == 'PREDICT':
        logging.info('All features done... sending dataframe over')
        return df


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

        feature_list = ['isFirstCap', 'Length', 'endY', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'POSNum', 'frontWord', 'backWord']
        vectorlist = []
        for i in range(24):
            vectorlist.append('WordVector' + str(i))
        feature_list = feature_list + vectorlist

        #takes the data and labels of the training and testing sets, and puts them in txt files
        data_train = train_df[feature_list].values
        label_train = train_df['TagNum'].values
        data_test = test_df[feature_list].values
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
        data_test = df[['isFirstCap', 'Length', 'endY', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'POSNum', 'frontWord', 'backWord']].values
        label_test = df['TagNum'].values
        np.savetxt(Dconfig.DATA_TEST_PATH, data_test)
        np.savetxt(Dconfig.LABEL_TEST_PATH, label_test)

        logging.info('Full evaluation split complete')

    #If the mode is set to all training
    elif mode == "TRAIN":

        #Takes all of the data and splits the data from label columns and saves them in txt files
        #for training
        data_train = df[['isFirstCap', 'Length', 'endY', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'POSNum', 'frontWord', 'backWord']].values
        label_train = df['TagNum'].values
        np.savetxt(Dconfig.DATA_TRAIN_PATH, data_train)
        np.savetxt(Dconfig.LABEL_TRAIN_PATH, label_train)

        logging.info('Full train split complete')

    #If the mode is set to a nonvalid mode
    else:
        logging.debug('Please put in a valid mode, or none to do a normal split')


#argparse code to allow command line functionality
def main():
    parser = argparse.ArgumentParser(description='Methods for feature generation and splitting of data')
    parser.add_argument("-c", metavar="<command>", help="'feature_gen' or 'split' or 'split_eval' or 'split_train'",)
    parser.add_argument("-f", metavar="filename", help="input file",)
    parser.add_argument("-m", metavar="mode", help="'EVAL', 'TRAIN', or BOTH",)
    parser.add_argument("-fgm", metavar="feature_gen mode", help="'EVAL' or None",)
    args = parser.parse_args()
    assert args.c in ['feature_gen', 'split'], "invalid parsing 'command'"

    if args.c == "feature_gen":
        if args.fgm == None:
            if args.f == None:
                feature_gen()
            else:
                feature_gen(filename = args.f)
        else:
            if args.f == None:
                feature_gen(mode = args.fgm)
            else:
                feature_gen(filename = args.f, mode = args.fgm)
        
    else:
        if args.m == None:
            data_split()
        elif args.m == 'EVAL' or args.m == 'TRAIN' or args.m == 'BOTH':
            data_split(mode = args.m)
        else:
            logging.info('Please enter a valid mode')

if __name__ == "__main__":
    main()
        





