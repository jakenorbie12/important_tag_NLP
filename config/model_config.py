#===============================================================================
#     Configurations for Model
#===============================================================================
PARAMETERS = {}
#Don't change these!
PARAMETERS['objective'] = 'multiclass'
PARAMETERS['num_classes'] = 17
#Change these!
PARAMETERS['learning_rate'] = 0.03
PARAMETERS['colsample_bytree'] = 0.7
PARAMETERS['max_depth'] = 20
PARAMETERS['min_split_gain'] = 0.3
PARAMETERS['n_estimators'] = 400
PARAMETERS['num_leaves'] = 100
PARAMETERS['reg_alpha'] = 1.1
PARAMETERS['reg_lambda'] = 1.3
PARAMETERS['subsample'] = 0.9
PARAMETERS['subsample_freq'] = 20

#Change these to edit the options available for optimal parameters
GRID_SEARCH_PARAM_GRID = {
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'subsample': [0.7, 0.8, 0.9],
}

GRID_SEARCH_USED = False
OPT_PARAMS = {}

if GRID_SEARCH_USED:
    PARAMETERS.update(OPT_PARAMS)


MODEL_PATH = './models/model.txt'


