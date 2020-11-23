#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import flask 

# In[3]:


app = flask.Flask(__name__, template_folder='templates')

df = pd.read_csv('./data/modeldata.csv')


# In[3]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[4]:


df = df.reset_index()
indices = pd.Series(df.index, index=df['name'])
all_restaurants = [df['name'][i] for i in range(len(df['name']))]
df['address']= df['address'].astype(str).str.replace(r'\[|\]|', '')
df['address']=df['address'].astype(str).str.replace(r'\'|\'|', '')


# In[6]:


def recommend(name):
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    resto_indices = [i[0] for i in sim_scores]
    resto = df['name'].iloc[resto_indices]
    add = df['address'].iloc[resto_indices]
    return_df = pd.DataFrame(columns = ['name', 'address'], dtype=str)
    return_df['name'] = resto
    return_df['address'] = add
    print(resto, add)
    return return_df


# In[ ]:


# set up main route

@app.route('/', methods = ['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        r_name = flask.request.form['restaurant_name']
        r_name = r_name.title()
        print(r_name)
        #check = difflib.get_close_matches(r_name, all_restaurants, cutout=0.50,n=1)
        if r_name not in all_restaurants:
            return(flask.render_template('negative.html', name=r_name))
        else:
            result_final = recommend(r_name)
            names = []
            addresses = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                addresses.append(result_final.iloc[i][1])
            # for name, address in result_final.iterrows():
            #     names.append(name)
            #     addresses.append(address)   
            return flask.render_template('positive.html', restaurant_names = names,
                                        restaurant_address=addresses, search_name = r_name)


if __name__ == '__main__':
    app.run(debug=True)





