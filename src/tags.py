#===============================================================================
#     Tags Runner
#===============================================================================
import argparse
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import sys
import logging
import dataset
ROOT_DIR = os.path.abspath("./config")
sys.path.append(ROOT_DIR)
import data_config as Dconfig
import model_config as Mconfig

logging.basicConfig(level=logging.INFO)


def train(model_path = Mconfig.MODEL_PATH):
        '''
        Takes the data and labels and trains a model
        +Input:
                model_path: the filepath to the model
        '''

        #Loads in the data and labels from the training set
        logging.info('Beginning model training...')
        data_train = np.loadtxt(Dconfig.DATA_TRAIN_PATH)
        label_train = np.loadtxt(Dconfig.LABEL_TRAIN_PATH)

        #Forms the data and labels into a dataset, and takes parameters
        #into a form that can be put into lightGBM
        d_train = lgb.Dataset(data_train, label=label_train)
        params = Mconfig.PARAMETERS

        #Using lightGBM, trains a model
        mod = lgb.train(params, d_train, 100)
        logging.info('Training complete. Saving to file.')

        #Saves the finished model into the filepath preset in the configs
        mod.save_model(model_path)
        logging.info('Finished')

def evaluate(mod_file = Mconfig.MODEL_PATH):
        '''
        Evaluates model by giving an accuracy score and confusion matrix
        +Input:
                model_path: the filepath to the model
        '''

        #Loads in the data, labels, and model
        logging.info('Beginning evaluation of model using data')
        data_test = np.loadtxt(Dconfig.DATA_TEST_PATH)
        label_test = np.loadtxt(Dconfig.LABEL_TEST_PATH)
        light_model = lgb.Booster(model_file = mod_file)

        #Uses the model to predict the data
        prediction_data = light_model.predict(data_test)

        #Changes the label to the most likely tag for each line
        classed_data = [np.argmax(line) for line in prediction_data]

        #Charts an accuracy score and makes a confusion matrix for the data
        #The accuracy is the percentage of the predicted tag correct divided by the actual tag
        logging.info('Complete. Generating accuracy score and confusion matrix')
        accuracy = accuracy_score(classed_data, label_test)
        cm = confusion_matrix(label_test, classed_data)
        print('\n')
        print("Accuracy = " + str(accuracy))
        print('\n')
        print('Confusion Matrix:')
        print(cm)

def predict(pred_file = Dconfig.PRED_DATASET_PATH):
        '''
        Uses the model to predict/identify the tags and saves the new data to a file
        +Input:
                pred_file: the filepath for the dataset to be predicted
        '''

        #Creates a featured Pandas dataframe from a csv file
        logging.info('Commensing prediction of data...')
        df = pd.read_csv(pred_file, sep='\t', encoding='unicode_escape')

        #Creates a copy of the dataframe for final usage
        df_final = df.copy()

        #Generates features onto the dataframe
        df = dataset.feature_gen(mode = 'PREDICT', data = df)
        logging.info('Done generating features')

        #Uses the features to create a viable dataset
        feature_list = ['isFirstCap', 'Length', 'endY', 'otherCap', 'endan',
                   'isNum', 'endS', 'endish', 'endese', 'propVow', 'POSNum', 'frontWord', 'backWord']
        vectorlist = []
        for i in range(24):
                vectorlist.append('WordVector' + str(i))
        feature_list = feature_list + vectorlist
        
        data = df[feature_list].values

        #Loads in the model
        light_model = lgb.Booster(model_file = Mconfig.MODEL_PATH)

        #Runs the data into the model, and changes the label to the most likely tag for each line
        predict_data = light_model.predict(data)
        classed_data = [np.argmax(line) for line in predict_data]
        logging.info('Prediction completed. Saving data to new file...')

        #Puts the tags onto the final dataframe, and saves it to a csv
        tag_file = open('./data/process/tag_array.txt', 'r')
        tag_array = tag_file.read()
        tag_array = tag_array.strip("[]")
        tag_array = tag_array.split(',')
        for idx, word in enumerate(tag_array):
                new_word = word.strip()
                new_word = new_word.strip("''")
                tag_array[idx] = new_word
        classed_data = [tag_array[line] for line in classed_data]
        df_final['Tag'] = classed_data
        df_final.to_csv(Dconfig.NEW_DATA_PATH)
        logging.info('Complete')

        
#argparse code to allow command line functionality
parser = argparse.ArgumentParser(description='Methods for Model Training, Evaluating and Predicting')
parser.add_argument("-c", metavar="<command>", help="'train', 'evaluate', or 'predict'",)
parser.add_argument("-f", metavar="file", help="input file",)
parser.add_argument("-mf", metavar="model file", help="model file path",)
args = parser.parse_args()

assert args.c in ['train', 'evaluate', 'predict'], "invalid parsing 'command'"

if args.c == "train":
        if args.mf == None:
                train()
        else:
                train(args.mf)
elif args.c == 'evaluate':
        if args.mf == None:
                evaluate()
        else:
                evaluate(args.mf)       
else:
        if args.f == None:
                predict()
        else:
                predict(args.f)
