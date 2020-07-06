#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import wheel
import setuptools
import scipy
import lightgbm as lgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
df_name = input("Enter the name of the file: ")
df = pd.read_csv(df_name, sep='\t', encoding='unicode_escape')


# In[55]:


#Functions shell

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

word_array = df.Word.unique().tolist()

def frontWord(x):
    if x > 0:
        return word_array.index(df['Word'][x-1])
    else:
        return
def backWord(x):
    if x < len(df.index) - 1:
        return word_array.index(df['Word'][x+1])
    else:
        return

array = df.Tag.unique().tolist()

def Tag2Num(x):
    return array.index(x)


# In[56]:


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

df['TagNum'] = df['Tag'].apply(lambda x: Tag2Num(x))

df['frontWord'] = df['Unnamed: 0'].apply(lambda x: frontWord(x))

df['backWord'] = df['Unnamed: 0'].apply(lambda x: backWord(x))


# In[69]:


sentences_group = df.groupby(['Sentence #'])
test_sentences = []
test_dfs = []
train_dfs = []
for i in range(750):
    found = False
    while found == False:
        num = random.randint(1, 2999)
        if not num in test_sentences:
            test_sentences.append(num)
            test_dfs.append(sentences_group.get_group(num))
            found = True
            
test_df = pd.concat(test_dfs)
drop_list = test_df['Unnamed: 0'].tolist()
train_df = df.copy().drop(drop_list)

data_train = train_df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
           'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
valid_train = train_df['TagNum'].values
data_test = test_df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
           'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
valid_test = test_df['TagNum'].values


# In[70]:


train_data = lgb.Dataset(data_train, label=valid_train)
parameters = {}
parameters['objective'] = 'multiclass'
parameters['num_class'] = 17
parameters['learning_rate'] = 0.03
d = lgb.train(parameters, train_data, 100)
#Save the model
d.save_model('model.txt')
y_pred = d.predict(data_test)
y_hat = [np.argmax(line) for line in y_pred]


# In[71]:


cm = confusion_matrix(valid_test, y_hat)
accuracy = accuracy_score(y_hat, valid_test)
f1 = f1_score(valid_test, y_hat, average = 'weighted')
print(accuracy)
print(f1)
print(cm)


# In[ ]:




