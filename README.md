# Important Tag Natural Language Processor

## Purpose
This contains various files pertaining to the generation of a lightGBM model designed to classify words by tags and also use the current model to predict data. The couple of files have names that are self-expanatory of their functions, but I will more thoroughly describe their use later in this readme.

## Project Description

The project consists of many files, but there are three important aspects:
1) Model Training: A lightGBM model can be made in order from a supervised and pre-tagged csv file. Such will create a model with numbers indicating each tag, numbering 0 to how ever many tags are included -1.
2) Execution: Testing of the model using pre-tagged and supervised data in order to get an accuracy score and a confusion matrix.
3) Impletmentation/Prediction: Use of a model to predict data, outputting a csv file with a new column indicating the number tag (Working on how to best change it to the string tag for each instance)

## Index

1. Dependencies
2. Installation
3. Usage of Files
4. Troubleshooting
5. Licensing

## Dependencies

I used anaconda and git to import and use all files. You semi-current versions and you should be fine. You can use others but I don't know how to accurately use them.
For the data, the program requires a csv in which the columns are titled "Sentence #", "Word", "POS", and potentially "Tag" and one column that records the index.
They should all be self-explanatory, but see Dataset_08-29-2019 in data/original if you are still confused.

## Installation

1) To download the github repository, simply navigate to the folder you want to use and type this command in git bash:
```
git clone https://github.com/jakenorbie12/important_tag_NLP.git
```

2) To install dependences, in anaconda navigate to the directory and type:
```
conda install --file installation/requirements.txt
```

3)You should be all good to go!

## Usage of Files

### Feature Generation and Data Splitting

1) In anaconda navigate to the folder using cd command and then type this:
```
python src/dataset.py feature_gen
```

2) This generates features for the data and saves it for a new file. For the data, put in the the data/original folder, and change
the filepath according in data_config.py file. Next, you split the data using this:
```
python src/dataset.py split
```

2a) Other versions of data splitting would be having all the data being training set and all the data being testing for evaluation purposes.
For complete training set use:
```
python src/dataset.py split_train
```
And for complete testing:
```
python src/dataset.py split_eval
```

### Training and Evaluating Data

1) Unless you plan to only evaluate the data (all testing). In anaconda navigate to the folder using cd and then type:
```
python src/tags.py train
```

2) From there you should have a model built, where unless you are only making a model type:
```
python src/tags.py evaluate
```

3) If you want to fine tune the parameters or have your own model, then you can change the filepath and parameters in the model_config script

### Predicting Data/Using the Model

1) For predicting data, you only need a csv with the word, part of speech, and a column named "Unnamed: 0" which is the number word.
Put the file into the data folder and change the filepath in data_config script.

2) Type this into anaconda:
```
ipython src/tags.py predict
```

2) Note: The data will be output into a new file named "New_Data.csv". To change this go to data_config and change it there.

## Troubleshooting

1) Make sure the config folders all point to the correct files.

2) Make sure the data is properly formatted.

For any issues you may have, please feel free to email me at jakenorbie@gmail.com.

## Licensing

Jake Norbie & Co.
