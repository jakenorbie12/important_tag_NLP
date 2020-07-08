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

### Model Generation

1) In anaconda navigate to the folder using cd and then type this:
```
python src/dataset.py feature_gen
```
2) Type in the file you want to use to generate the model. 

3) This uses lightGBM to create a Model and saves it under "model.txt". If you want to change the name go to line 162 or under the comment that says "#Save the model".

4) Also, it is important to note that an accuracy score, F1 score, and confusion matrix will show unless commented out.

### Testing Data

1) In anaconda navigate to the folder using cd and then type:
```
python src/predict.py evaluate
```
2) Type in the file you want to use to test the data.

3) Using the model, it tests the data against the true data to show the accuracy and confusion matrix.

### Predicting Data/Using the Model

1) In anaconda type in:
```
ipython src/predict.py predict
```
2) Type in the name of the file you want to use

3) Note: The data will be output into a new file named "New_Data.csv". To change this go to line 101 or the comment "#Sets the name for the new file" and change it.

## Troubleshooting

For any issues you may have, please feel free to email me at jakenorbie@gmail.com.

## Licensing

Jake Norbie & Co.
