#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


# importing the necessary libraries 

# In[2]:


movies=pd.read_csv(r'C:\Users\parih\OneDrive\Desktop\movie_rec\Data\movies11.csv')
rating=pd.read_csv(r'C:\Users\parih\OneDrive\Desktop\movie_rec\Data\ratings2.csv')


# loading movies and ratings data frames

# In[3]:


movies.info()
movies.head(10)
movies.drop('genres', axis=1, inplace=True)


# In[4]:


rating.info()
rating


# I plotted a graph to analyse movie vs rating_count to filter movies with less ratings, and fixed minimum 100 ratings from users to consider it into final data frame.

# In[5]:


rating_count = rating['movieId'].value_counts()
rating_count

p_movie = rating_count[rating_count>100].index
p_movie

df= rating[rating.movieId.isin(p_movie)]
#df
p_movie_df  = movies[movies['movieId'].isin(p_movie)]
p_movie_df


# In[6]:


f,ax = plt.subplots(1,1,figsize=(12,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(rating_count.index,rating_count,color='Blue')
plt.axhline(y=100,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


# In[7]:


movie_count=df['userId'].value_counts()
valid_user= movie_count[movie_count>100].index
df=df[df.userId.isin(valid_user)]
df


# In[8]:


f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(movie_count.index,movie_count,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()


# And to filter out users, I removed users who rated less than 50 movies.

# In[9]:


df1=df.copy()


# In[10]:


new_df = df.pivot(index='movieId',columns='userId',values='rating')


# Creating a matrix with userId as columns and movieId as rows.

# In[11]:


new_df


# In[12]:


new_df.fillna(0, inplace=True)
#new_df1=new_df.reset_index()
new_df


# Replacing NaN with 0, so that we can perform different operations on the matrix

# In[13]:


movie_id_map = {movie_id: idx for idx, movie_id in enumerate(new_df.index)}
index_movie_map = {idx: movie_id for movie_id, idx in movie_id_map.items()}
reversed_mapping = {v: k for k, v in index_movie_map.items()}
#reversed_mapping[205383]


# We can observe in the matrix that movieId are not continuous and while using the matrix will not consider the actual movieId, instead it consider it position.\
# So we are mapping the movie_id and Index position to overcome this problem.

# In[14]:


sparcity = 1-(np.count_nonzero(new_df)/float(new_df.size))
print(sparcity)


# We can observe in our matrix that there are lot of null (0) values, which affect the training of our model, and to estimate the number of null values I calculated sparcity.

# In[15]:


csr_df = csr_matrix(new_df.values)


# To overcome the high sparcity problem I am using CSR Matrix. 

# In[16]:


csr_df


# In[17]:


model = NearestNeighbors(metric='cosine',algorithm='brute',n_jobs=-1,n_neighbors=15)


# I am using **NearestNeighbour** model with tuning some hyperparameters to get the best output.

# In[18]:


model.fit(csr_df)


# Fitting csr data frame into our model.

# In[19]:


val =(204698,5.22) 
movie = movies.loc[movies['movieId'] == val[0], 'title']
movie.values[0]


# I was just verifying the output of this code, ignore this(hehe).

# In[20]:


def get_movie_recommendation(movie_id, k=10):
   
        movie_idx = reversed_mapping[movie_id]
        #print(movie_idx)
        print(f"selected movie is : {p_movie_df.iloc[movie_idx]['title']}")
        distances, indices = model.kneighbors(csr_df[movie_idx], n_neighbors=k+1)
    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        #print(rec_movie_indices)
        recommend_frame = []
        
        for val in rec_movie_indices:
            id= val[0]
            recommend_frame.append({'Title':movies.iloc[val[0]]['title'],'Distance':val[1]})
        df4 = pd.DataFrame(recommend_frame,index=range(1,10+1))
        #print(df4)
        return df4 
        


# This is our final function which give required output(recommend related movies)\
# This takes movie_id as input and recommend k movies.\
# At first this get the position of the input movie in our matrix using mapping data\
# Then using model it calculates k nearest points,(with distance and movie_id of that point)\
# Then we make a list of our output\
# And make a dataframe consisting of movie_title and distance from input movie_id.

# In[21]:


movie_list = p_movie_df[p_movie_df['title'].str.contains('Spider-Man')]
#len(movie_list): 
movie_id= movie_list.iloc[0]['movieId']
#print(movie_id)
get_movie_recommendation(movie_id)    
        


# This cell can be used to get recommendation, we can change movie_name in the 1st line of this cell and get output from the built function.

# In[27]:


def recommend (movie_name):
    try :   
        movie_list = p_movie_df[p_movie_df['title'].str.contains(movie_name)]
        #len(movie_list): 
        movie_id= movie_list.iloc[0]['movieId']
        #print(movie_id)
    
        result = get_movie_recommendation(movie_id)
        return result
    except IndexError as e :
        print(f"{movie_name} is not in data set, please check spelling and try again and make sure first letter is capital in your movir name")

    pass    


# Above cell gives error and complete detail about error when we give wrong movie_name\
# So I created another function to reduce the ugliness of this notebook when error occur.\
# And we do not need to find and change movie_name in the above code, we can just write **recommend("type movie name here")** and we get our output.

# In[28]:


recommend("Joker")


# In[24]:


#please remove hash in below 2 lines before running this code
#movie_name = input("enter movie name")
#recommend(movie_name)


# Above cell just reduce our work type recommend("movie name"), we can just run the code and it ask us for input(movie_name)\
# then it gives output from the function.

# In[25]:


import pickle


# In[26]:


pickle.dump(p_movie_df.to_dict(),open('movies.pkl','wb'))


# In[ ]:





