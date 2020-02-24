'''
Requirements :
You need to have run q4PreProcessing_notebook.ipynb before running this code. 
(in anaconda prompt)
    Jupyter notebook -> conda install -c conda-forge notebook

    
In this iteration of the code we take into account:
    1) user_Id
    2) ranking results now take into account the user's rating of a movie (if available)
    3) we cluster users together based on how and which movies they rate. Then we avg out the rating per cluster, thus avoiding some NaNs in user_rating
    4) We train a Ranfom Forest Classifier on each user trained on tf-idf data from the title in movies they have rated and the on hot encoding of that movie's genre.
       Then we used said classifier to predict the rating on the rest of the movies. ( This gives a very bad result and should not be followed, it is only done for demonstrative
       purposes, to get some exercise on the pipeline of ML models in an Information Retrival setting)
'''
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import pickle


def queryInput(user_input):
    ''' returns the movie_Id, title, BM25 score '''
    # ps = per_search since BM25 _score changes depending on user_input
    result_params_list_df_ps = pd.DataFrame(columns=['movieId', 'title', 'genres', 'BM25_score'])

    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(user_input)}}}, size = 1000)
    hits = res['hits']['total']['value']
    print("Got {} Hits:".format(hits))

    # I set the maximum number of results to 1000. The default was 10. So we need to take that into account
    try :
        for i in range(hits):
            temp_df = pd.DataFrame([ [ int(res['hits']['hits'][i]['_source']['movieId']), res['hits']['hits'][i]['_source']['title'], res['hits']['hits'][i]['_source']['genres'], res['hits']['hits'][i]['_score'] ] ], columns=['movieId', 'title', 'genres', 'BM25_score'])
            result_params_list_df_ps = result_params_list_df_ps.append(temp_df, ignore_index = True)
    except:
        for i in range(1000):
            temp_df = pd.DataFrame([ [ int(res['hits']['hits'][i]['_source']['movieId']), res['hits']['hits'][i]['_source']['title'], res['hits']['hits'][i]['_source']['genres'], res['hits']['hits'][i]['_score'] ] ], columns=['movieId', 'title', 'genres', 'BM25_score'])
            result_params_list_df_ps = result_params_list_df_ps.append(temp_df, ignore_index = True)
    return  result_params_list_df_ps


def preProcessing():
    '''Returns the AVG rating of each movie (movie_avg_rating_df)
    and user rating per movie (movie_rating_df_pu)
    Also, loads the data from queryIndexQ3 where we clustered the users to fill NaN'''
    rating_df = pd.read_csv('./data/ratings.csv') 
    #dataframe containing user ratings after filling most NaN base on user clustering (look at queryIndexQ3.py)
    user_ratings_after_clustering = pickle.load( open("./data/user_ratings_after_clustering.p", "rb")) 
    #pu = per user
    movie_rating_df_pu = rating_df[['userId','movieId','rating']]
    movie_avg_rating_df = movie_rating_df_pu.groupby(by='movieId').mean()
    movie_avg_rating_df = movie_avg_rating_df.drop('userId', axis=1).reset_index()
    
    return movie_avg_rating_df, user_ratings_after_clustering


def startLoop(movie_avg_rating_df, user_ratings_after_clustering):
    ''' Basically the main function of the program. 
    Takes any pre-processed data as input '''

    print('type "//exit" if you want to exit the search')
    user_input_movie = input("Which movie do you want? (by title): \n")
    
    while( user_input_movie != '//exit' ) :
        
        user_input_user = input("For which user do you want to search? (int): \n")
        while ( ( user_input_user.isdigit() == False) ):
            user_input_user = input("For which user do you want to search? (int): \n")

        #get ranking from Elasticsearch
        query_result_params_df = queryInput(user_input_movie)
        #compute final ranking based on additional data ( movie_avg_rating_df, movie_rating_df_pu )
        final_df = finalRanking(query_result_params_df, movie_avg_rating_df, user_ratings_after_clustering, user_input_user)
        #print results
        print(final_df)


        print('type "//exit" if you want to exit the search')
        user_input_movie = input("Which movie do you want? (by title): \n")
    return


def finalRanking(query_result_params_df, movie_avg_rating_df, movie_rating_df_pu, user_input_user):
    ''' computes the final_df which holds the final ranking '''
    final_df = query_result_params_df.copy(deep=True)

    #add avg_rating
    final_df = final_df.merge(movie_avg_rating_df, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'avg_rating'}, inplace=True)
    
    #create a temp DataFrame so no changes are made to the original 'movie_rating_df_pu'
    temp = movie_rating_df_pu.copy(deep=True)
    #drop all rows from the user ratings where the user is not the one we want
    temp.drop(temp[temp['userId'] != int(user_input_user)].index, inplace=True)
    #add user_rating  ( after clustering this time )
    final_df = final_df.merge(temp, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'user_rating_after_clustering'}, inplace=True)

    #now add the data from the jupyter notebook where we fille user NaN ratings based on tf-idf vactors and a RandomForestClassifier
    user_predicted_data = pickle.load( open("./data/predicted_result/user_{}_predicted_data.p".format(user_input_user), "rb")) 
    temp = user_predicted_data.copy(deep=True)
    final_df = final_df.merge(temp, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'user_rating_after_RFC'}, inplace=True)

    #for the sake of simplicity our similarity algorithm will be a linear combinationi of the above 4 scores
    final_df['final_score'] = np.nan
    # Note that user_rating_after_RFC is in all passes because it has no Null Predictions. A prediction was made for all users for every movie
    #first pass 
    final_df['final_score'] = final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating_after_clustering'] + final_df['user_rating_after_RFC']
    #second pass for the cases where the user has not added a rating for the movie ( as far as the clustering is concerned )
    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating_after_RFC'], inplace=True) 
    #third pass where there is no avg_rating for a movie so we only keep the BM25 score
    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['user_rating_after_RFC'], inplace=True) 
    
    #Sort the dataframe based on the final_score
    final_df = final_df.sort_values(by = 'final_score', ascending=False).reset_index()
    final_df.drop(['index','genres','userId'], axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)

    return final_df


#######################################################

if __name__ == "__main__":
    # start Elasticsearch server
    es = Elasticsearch()
    # compute avg rating for all movies as a pre processing step
    movie_avg_rating_df, user_ratings_after_clustering = preProcessing()
    # get to main loop
    startLoop(movie_avg_rating_df, user_ratings_after_clustering)