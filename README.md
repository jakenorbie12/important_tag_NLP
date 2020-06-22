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

2) Everything should be there!

## Usage of Files

### Model Generation

1) In anaconda navigate to the folder using cd and then type this:
```
ipython NLP_mod_gen.py
```

## Troubleshooting

For any issues you may have, please feel free to email me at jakenorbie@gmail.com.

## Licensing

Jake Norbie & Co.
