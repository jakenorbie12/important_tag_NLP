#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn as sk
import lightgbm as lgb
df_name = input("Enter the name of the file: ")
df_name_orig = input("Enter the name of the file used for ")
df = pd.read_csv(df_name, sep='\t', encoding='unicode_escape')
df_new = df.copy()
# Dataset_08-29-2019.txt


# In[3]:


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

tagArray = df.Tag.unique().tolist()
def Tag2Num(x):
    return tagArray.index(x)


# In[4]:


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

df['frontWord'] = df['Unnamed: 0'].apply(lambda x: frontWord(x))

df['backWord'] = df['Unnamed: 0'].apply(lambda x: backWord(x))


# In[7]:


data_test = df[['isFirstCap', 'Length', 'endY', 'isNNP', 'isJJ', 'isCD', 'otherCap', 'endan',
           'isNum', 'endS', 'endish', 'endese', 'propVow', 'frontWord', 'backWord']].values
light_model = lgb.Booster(model_file = 'model.txt')
y_pred = light_model.predict(data_test)
y_hat = [np.argmax(line) for line in y_pred2]


# In[11]:


df_new['Tag'] = y_hat
df_new.to_csv('New_Data.csv', encoding = 'unicode-escape')


# In[ ]:




