#===============================================================================
#     Data Transformation and Splitting
#===============================================================================
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import os
import sys
import logging
ROOT_DIR = os.path.abspath("./config")
sys.path.append(ROOT_DIR)
import model_config as Mconfig
import data_config as Dconfig
import argparse


logging.basicConfig(level=logging.INFO)

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=5, scoring_fit='accuracy',
                       do_probabilities = False):
    '''
    Runs every parameter configuration from param_grid and finds the optimal
        parameters
    +Inputs:
        X_train_data: the training data
        X_test_data: the testing set data
        y_train_data: the labels for the training data
        y_test_data: the labels for the testing set data
        model: the model used
        param_grid: the dictionary of potential parameters and values
        cv: the number of k-folds
        scoring_fat: method of measuring accuracy
        do_probabilities: whether the algorithm will use probabilities
    '''

    logging.info('Beginning testing of parameters')

    #Performs a grid search using the arguments given
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )

    
    #Fits the best model
    fitted_model = gs.fit(X_train_data, y_train_data)

    #Does probabilities or not depending on the argument 'do_probabilities' passed
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)

    #Returns the fitted model and predicted labels
    logging.info('Completed testing, now fitting and returning model and pred')
    return fitted_model, pred

def opt_mod(param_path = Mconfig.GRID_SEARCH_PARAM_GRID):
    '''
    Uses the algorithm pipeline function to optimize the parameters on the lightGBM model
    +Inputs:
        param_path: the file path to the params for the grid search
    '''

    logging.info('Loading model, parameters, and data')
    
    #Preps the model and param_grid
    model = lgb.LGBMClassifier()
    param_grid = param_path

    #Loads in the data from the data split
    data_train = np.loadtxt(Dconfig.DATA_TRAIN_PATH)
    label_train = np.loadtxt(Dconfig.LABEL_TRAIN_PATH)
    data_test = np.loadtxt(Dconfig.DATA_TEST_PATH)
    label_test = np.loadtxt(Dconfig.LABEL_TEST_PATH)

    #Runs the algorithm_pipeline function, returning the predictive labels and model
    model, pred = algorithm_pipeline(data_train, data_test, label_train, label_test, model, 
                                 param_grid, cv=5, scoring_fit='accuracy')


    #Prints the best score and parameters fields
    logging.info('Now showing optimal parameters: ')
    logging.info(model.best_params_)
    logging.info('---------------------------------')

    logging.info('Now showing best score: ')
    logging.info(model.best_score_)
    logging.info('---------------------------------')

    logging.info('Writing in optimal parameters')
    f = open('./config/model_config.py', 'r')
    lines = f.readlines()
    f.close()
    f = open('./config/model_config.py', 'w')
    for line in lines:
        if 'GRID_SEARCH_USED = ' in line:
            f.write('GRID_SEARCH_USED = True\n')
        elif 'OPT_PARAMS =' in line:
            f.write('OPT_PARAMS = ' + str(model.best_params_) + '\n')
        else:
            f.write(line)      
    f.close()


#argparse code to allow command line functionality
parser = argparse.ArgumentParser(description='Hypertesting Parameters Function')
parser.add_argument("-c", metavar="<command>", help="'opt_mod'",)
parser.add_argument("-pf", metavar="parameters file", help="parameters file path",)

args = parser.parse_args()
assert args.c in ['opt_mod'], "invalid parsing 'command'"

if args.c == 'opt_mod':
    if args.pf == None:
        opt_mod()
    else:
        opt_mod(args.pf)
else:
    logging.info('Please enter a valid command')





