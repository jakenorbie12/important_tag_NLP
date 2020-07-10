#===============================================================================
#     Tags Runner
#===============================================================================
import argparse
import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import sys
import logging
import functions
ROOT_DIR = os.path.abspath("./config")
sys.path.append(ROOT_DIR)
import data_config as Dconfig
import model_config as Mconfig

logging.basicConfig(level=logging.INFO)


def train():
        '''
        Takes the data and labels and trains a model
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
        mod.save_model(Mconfig.MODEL_PATH)
        logging.info('Finished')

def evaluate():

        #Loads in the data, labels, and model
        logging.info('Beginning evaluation of model using data')
        data_test = np.loadtxt(Dconfig.DATA_TEST_PATH)
        label_test = np.loadtxt(Dconfig.LABEL_TEST_PATH)
        light_model = lgb.Booster(model_file = Mconfig.MODEL_PATH)

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

def predict():

        #Creates a featured Pandas dataframe from a csv file
        logging.info('Commensing prediction of data...')
        df = pd.read_csv(Dconfig.PRED_DATASET_PATH, sep='\t', encoding='unicode_escape')

        #Creates a copy of the dataframe for final usage
        df_final = df.copy()

        #Generates features onto the dataframe
        df = functions.feature_gen_4_pred()

        #Uses the features to create a viable dataset
        data = df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
           'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values

        #Loads in the model
        light_model = lgb.Booster(model_file = Mconfig.MODEL_PATH)

        #Runs the data into the model, and changes the label to the most likely tag for each line
        predict_data = light_model.predict(data)
        classed_data = [np.argmax(line) for line in predict_data]
        logging.info('Prediction completed. Saving data to new file...')

        #Puts the tags onto the final dataframe, and saves it to a csv
        df_final['Tag'] = classed_data
        df_final.to_csv(Dconfig.NEW_DATA_PATH, encoding = 'unicode-escape')

        
#argparse code to allow command line functionality
parser = argparse.ArgumentParser(description='Methods for Model Training, Evaluating and Predicting')
parser.add_argument("command", metavar="<command>", help="'train', 'evaluate', or 'predict'",)
args = parser.parse_args()

assert args.command in ['train', 'evaluate', 'predict'], "invalid parsing 'command'"

if args.command == "train":
        train()
elif args.command == 'evaluate':
        evaluate()       
else:
        predict()
