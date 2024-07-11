import requests
import streamlit as st
import pickle
import pandas as pd
import nbformat
from IPython import get_ipython
#from mymodel import recommend
#from Mymodel import get_movie_recommendation,recommend
#st.title('MOVIE RECOMMENDATION SYSYTEM')

movies_dict = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)

map = pickle.load(open('map.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
csr=pickle.load(open('csr.pkl','rb'))

st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
"Type or select a movie from the dropdown",
 movies['title'].values
)

def get_movie_recommendation(movie_id, k=10):
   
        movie_idx = map[movie_id]
        #print(movie_idx)
        print(f"selected movie is : {movies.iloc[movie_idx]['title']}")
        distances, indices = model.kneighbors(csr[movie_idx], n_neighbors=k+1)
    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        #print(rec_movie_indices)
        recommend_frame = []
        
        #for val in rec_movie_indices:
        #    id= val[0]
        #    recommend_frame.append({'Title':movies.iloc[val[0]]['title'],'Distance':val[1]})
        #df4 = pd.DataFrame(recommend_frame,index=range(1,10+1))
        #print(df4)


        for i in rec_movie_indices:
             recommend_frame.append(movies.iloc[i[0]].title)
        return recommend_frame



        #return df4 



def recommend (movie_name):
    try :   
        movie_list = movies[movies['title'] == movie_name]
        #len(movie_list): 
        movie_id= movie_list.iloc[0]['movieId']
        #print(movie_id)
    
        result = get_movie_recommendation(movie_id)
        return result
    except IndexError as e :
        print(f"{movie_name} is not in data set, please check spelling and try again and make sure first letter is capital in your movir name")


is_clicked = st.button(label="Get Recommendation")


if is_clicked:
     r=recommend(selected_movie_name)
     st.header("Recommende movies are:")

     for i in r:
        st.write(i)






