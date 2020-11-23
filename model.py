#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from rake_nltk import Rake
from nltk.corpus import stopwords 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import flask 
import difflib


# In[2]:


app = flask.Flask(__name__, template_folder='templates')

df = pd.read_csv('./data/modeldata.csv')


# In[3]:


count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[4]:


df = df.reset_index()
indices = pd.Series(df.index, index=df['name'])


# In[5]:


all_restaurants = [df['name'][i] for i in range(len(df['name']))]


# In[6]:


def recommend(name, cosine_sim=cosine_sim):
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    resto_indices = [i[0] for i in sim_scores]
    resto = df['name'].iloc[resto_indices]
    add = df['address']
    return_df = pd.DataFrame(columns = ['name', 'address'])
    return_df['name'] = resto
    return_df['address'] = add
    return return_df


# In[ ]:




