'''
Requirements :
    ( you could download Anaconda distribution and have all these included )
    Pandas library -> pip install pandas  
    numpy library -> pip install numpy 
    sklearn library ->
    Scikit-learn requires:
        Python (>= 3.5)
        NumPy (>= 1.11.0)
        SciPy (>= 0.17.0)
        joblib (>= 0.11)
        1) pip install Cython
        2) pip install scipy
        3) pip install matplotlib
        4) pip install -U scikit-learn
    if using Visual Studio you need to download -> "Microsoft Visual C++ Build Tools": https://visualstudio.microsoft.com/downloads/

In this iteration of the code we take into account:
    1) user_Id
    2) ranking results now take into account the user's rating of a movie (if available)
    3) we cluster users together based on how and which movies they rate. Then we avg out the rating per cluster, thus avoiding some NaNs in user_rating
'''
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances
import warnings


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
    and user rating per movie (movie_rating_df_pu)'''
    print('\nPlease wait while we do some pre-processing.....\n')
    rating_df = pd.read_csv('./data/ratings.csv')  
    #pu = per user
    movie_rating_df_pu = rating_df[['userId','movieId','rating']]

    #In this iteration we cluster users together and avg out missing rating according to the cluster they belong to
    clustered_users_df = clusterUsers(movie_rating_df_pu)
    filled_user_ratings_df = fill_user_ratings(clustered_users_df)   
    #get the avg movie rating from actual user ratings
    movie_avg_rating_df = movie_rating_df_pu.groupby(by='movieId').mean()
    movie_avg_rating_df = movie_avg_rating_df.drop('userId', axis=1).reset_index()
    return movie_avg_rating_df, filled_user_ratings_df


def startLoop(movie_avg_rating_df, movie_rating_df_pu):
    ''' Basically the main function of the program. 
    Takes any pre-processed data as input '''

    print('type "exit" if you want to exit the search')
    user_input_movie = input("Which movie do you want? (by title): \n")
    
    while( user_input_movie != 'exit' ) :
        
        user_input_user = input("For which user do you want to search? (int): \n")
        while ( ( user_input_user.isdigit() == False) ):
            user_input_user = input("For which user do you want to search? (int): \n")

        #get ranking from Elasticsearch
        query_result_params_df = queryInput(user_input_movie)
        #compute final ranking based on additional data ( movie_avg_rating_df, movie_rating_df_pu )
        final_df = finalRanking(query_result_params_df, movie_avg_rating_df, movie_rating_df_pu, user_input_user)
        #print results
        print(final_df)


        print('type "exit" if you want to exit the search')
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
    #add user_rating
    final_df = final_df.merge(temp, on = 'movieId', how = 'left')
    final_df.rename(columns = {'rating':'user_rating'}, inplace=True)

    #for the sake of simplicity our similarity algorithm will be a linear combinationi of the above 3 scores
    final_df['final_score'] = np.nan
    #first pass 
    final_df['final_score'] = final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating']
    #second pass for the cases where the user has not added a rating for the movie
    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['avg_rating'], inplace=True)
    
    #Sort the dataframe based on the final_score
    final_df = final_df.sort_values(by = 'final_score', ascending=False).reset_index()
    final_df.drop('index', axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)

    return final_df


def fill_user_ratings(clustered_users_df):
    ''' Fills any NaN rating accoding to mean rating of the movie in the cluster the user belongs to.
    SOS : Keep in mind that we will still have NULL values that are due to the fact that some movies are still not rated. 
    Either withing the users of the cluster or across all users in our dataset '''
    #make a deepcopy so clusterd_users_df does not change
    filled_user_ratings_df = clustered_users_df.copy(deep=True)

    for cluster in filled_user_ratings_df['cluster'].unique():
        #we need the temp so that .mean() works correctly ( within the cluster only )
        temp = filled_user_ratings_df[filled_user_ratings_df['cluster'] == cluster]
        temp.fillna(temp.mean(), inplace=True)
        #then we overide the data
        filled_user_ratings_df[filled_user_ratings_df['cluster'] == cluster] = temp

    #now we need to pivot back to the format we had before clustering.... ( so we can merge in final_df )
    filled_user_ratings_df = filled_user_ratings_df.reset_index().drop('cluster', axis=1).melt('userId', var_name='movieId', value_name='rating').sort_values(by=['userId','movieId'])
    return filled_user_ratings_df


def combine_with_cluster(df, cluster_labels):
    '''combines the labels from KMeans with the Dataframe df'''
    df['cluster'] = pd.Series(cluster_labels, index=df.index)
    return 

def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
    # 2- sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    # 3- slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies


def clusterUsers(movie_rating_df_pu):
    '''This function clusters the users together so we can fill any NaN values is user_rating
    note: I did not use Standard Scaler because all dimensions represent movies and the values are user ratings.
    Thus, the distance in every dimension is in the same metric. No need to scale or normalize.'''
    X = movie_rating_df_pu.pivot(index='userId', columns='movieId', values='rating')
    #get 1000 most rated movies
    X_best = get_most_rated_movies(X,1000)
    #any NaN values are filled with the avg rating of the movie according to the rest of the users
    X_best_noNaN = X_best.fillna(0)
    #run Kmeans and get final df
    predictions = predictWithKmeans(15, X_best_noNaN)   
    #combine predictions with initial dataframe X that contains all user-movie-ratings
    combine_with_cluster(X, predictions)
    return X

def predictWithKmeans(clusters, sparce_matrix):
    '''This functions clusters any data in the sparce_matrix using KMEANS with k=clusters.'''
    return KMeans(n_clusters = clusters, algorithm='full').fit_predict(sparce_matrix)


#######################################################

if __name__ == "__main__":
    #I am ignoring a warining that appears and does not affect the code. User does not need to see it
    warnings.simplefilter("ignore")

    # start Elasticsearch server
    es = Elasticsearch()
    # compute avg rating for all movies as a pre processing step
    movie_avg_rating_df, movie_rating_df_pu = preProcessing()
    # get to main loop
    # remember movie_rating_df_pu now is AFTER we have clustered the users and filled most NaNs
    startLoop(movie_avg_rating_df, movie_rating_df_pu)

