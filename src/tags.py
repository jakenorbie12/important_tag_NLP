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

        logging.info('Beginning model training...')
        data_train = np.loadtxt(Dconfig.DATA_TRAIN_PATH)
        label_train = np.loadtxt(Dconfig.DATA_TRAIN_PATH)

        d_train = lgb.Dataset(data_train, label=label_train)
        params = Mconfig.PARAMETERS
        mod = lgb.train(params, d_train, 100)
        logging.info('Training complete. Saving to file.')

        mod.save_model(Mconfig.MODEL_PATH)
        logging.info('Finished')

def evaluate():

        logging.info('Beginning evaluation of model using data')
        data_test = np.loadtxt(Dconfig.DATA_TRAIN_PATH)
        label_test = np.loadtxt(Dconfig.DATA_TRAIN_PATH)
        light_model = lgb.Booster(model_file = Mconfig.MODEL_PATH)

        prediction_data = light_model.predict(data_test)
        classed_data = [np.argmax(line) for line in prediction_data]
        logging.info('Complete. Generating accuracy score and confusion matrix')
        accuracy = accuracy_score(classed_data, label_test)
        cm = confusion_matrix(label_test, classed_data)
        print("Accuracy = " + str(accuracy))
        print('\n')
        print('Confusion Matrix:')
        print(cm)

def predict():

        logging.info('Commensing prediction of data...')
        df = functions.feature_gen_4_pred()
        df_new = df.copy()
        data = df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
           'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
        light_model = lgb.Booster(model_file = Mconfig.MODEL_PATH)
        
        predict_data = light_model.predict(data)
        classed_data = [np.argmax(line) for line in predict_data]

        logging.info('Prediction completed. Saving data to new file...')
        df_new['Tag'] = classed_data
        df_new.to_csv(Dconfig.NEW_DATA_PATH, encoding = 'unicode-escape')

        

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
