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
from itertools import product
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
    and user rating per movie (movie_rating_df_pu)'''
    print('\nPlease wait while we do some pre-processing.....')
    rating_df = pd.read_csv('./data/ratings.csv')  
    #pu = per user
    movie_rating_df_pu = rating_df[['userId','movieId','rating']]

    #get the avg movie rating from actual user ratings
    movie_avg_rating_df = movie_rating_df_pu.groupby(by='movieId').mean()
    movie_avg_rating_df = movie_avg_rating_df.drop('userId', axis=1).reset_index()
    
    #In this iteration we cluster users together and avg out missing rating according to the cluster they belong to
    clustered_users_df = clusterUsers(movie_rating_df_pu,movie_avg_rating_df)
    filled_user_ratings_df = fillUserRatings(clustered_users_df) 

    #let's save filled_user_rating_df so we don't have to re-run all the clustering in the next script
    print("\nSaving the clustering result...")
    pickle.dump(filled_user_ratings_df, open("./data/user_ratings_after_clustering.p", "wb"))

    return movie_avg_rating_df, filled_user_ratings_df


def startLoop(movie_avg_rating_df, movie_rating_df_pu):
    ''' Basically the main function of the program. 
    Takes any pre-processed data as input '''

    print('\ntype "//exit" if you want to exit the search')
    user_input_movie = input("Which movie do you want? (by title): ")
    
    while( user_input_movie != '//exit' ) :
        
        user_input_user = input("\nFor which user do you want to search? (int): ")
        while ( ( user_input_user.isdigit() == False) ):
            user_input_user = input("\nFor which user do you want to search? (int): ")

        #get ranking from Elasticsearch
        query_result_params_df = queryInput(user_input_movie)
        #compute final ranking based on additional data ( movie_avg_rating_df, movie_rating_df_pu )
        final_df = finalRanking(query_result_params_df, movie_avg_rating_df, movie_rating_df_pu, user_input_user)
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

    #for the sake of simplicity our similarity algorithm will be a linear combinationi of the above 3 scores
    final_df['final_score'] = np.nan
    #first pass 
    final_df['final_score'] = final_df['BM25_score'] + final_df['avg_rating'] + final_df['user_rating_after_clustering']
    #second pass for the cases where the user has not added a rating for the movie
    final_df['final_score'].fillna(final_df['BM25_score'] + final_df['avg_rating'], inplace=True)
    #third pass where there is no avg_rating for a movie so we only keep the BM25 score
    final_df['final_score'].fillna(final_df['BM25_score'], inplace=True)
    
    #Sort the dataframe based on the final_score
    final_df = final_df.sort_values(by = 'final_score', ascending=False).reset_index()
    final_df.drop(['index','genres','userId'], axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)

    return final_df


def fillUserRatings(clustered_users_df):
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


def combineWithCluster(df, cluster_labels):
    '''combines the labels from KMeans with the Dataframe df'''
    df['cluster'] = pd.Series(cluster_labels, index=df.index)
    return 


def cartesianProduct(movie_rating_df_pu):
    '''This function creates a dataframe with all the user-movie combinations'''
    l1 = list(movie_rating_df_pu['userId'].unique())
    l2 = list(movie_rating_df_pu['movieId'].unique())
    temp = pd.DataFrame(list(product(l1, l2)), columns=['userId', 'movieId'])
    temp.sort_values(by=['userId','movieId']).reset_index(inplace=True, drop=True)
    temp = temp.merge(movie_rating_df_pu, on = ['userId','movieId'], how='left')
    return temp


def fillNanWithAvgGenreRating(movie_rating_df_pu, movie_avg_rating_df):
    ''' This function fills NaN values based on the average rating of each unique genre combination '''
    #create a df that holds all user-movie combinations
    user_movie_product_df = cartesianProduct(movie_rating_df_pu)

    #create the avg rating per genre 
    movie_details_df = pd.read_csv('./data/movies.csv')
    avg_rating_per_genre = movie_avg_rating_df.merge(movie_details_df[['movieId','genres']], on='movieId', how='left')
    avg_rating_per_genre.drop('movieId',axis=1,inplace=True)
    avg_rating_per_genre = avg_rating_per_genre.groupby(by='genres').mean()
    avg_rating_per_genre.rename(columns={'rating':'avg_rating_per_genre'},inplace=True)
    
    #fill NaN based on above
    movie_rating_df_pu_with_genre = user_movie_product_df.merge(movie_details_df[['movieId','genres']], on='movieId', how='left')
    movie_rating_df_pu_with_genre = movie_rating_df_pu_with_genre.merge(avg_rating_per_genre, on='genres',how='left')
    movie_rating_df_pu_with_genre['rating'] = movie_rating_df_pu_with_genre.rating.fillna(movie_rating_df_pu_with_genre.avg_rating_per_genre)
    movie_rating_df_pu_noNaN = movie_rating_df_pu_with_genre.drop(['avg_rating_per_genre','genres'], axis=1)

    return movie_rating_df_pu_noNaN

def getMostRatedMovieColumns(user_movie_ratings, max_number_of_movies):
    '''This function takes in the user's movie rating and a paramater that tells it how many 
    of the most rated (by number of ratings) movies to keep. Then it returns the movieId of those movies
    which happens to be the column name after we have pivoted the dataframe in another function.'''
    #create a deepcopy of user's movie ratings and count how many exist for each movie
    temp = user_movie_ratings.copy(deep=True)
    #add that value at the end of the temp dataframe
    temp = temp.append(user_movie_ratings.count(), ignore_index=True)
    #icrease the index by 1 so that userId remains the same
    temp.index = range(1,len(temp)+1)
    #sort the temp dataframe according to the appended values
    temp_sorted = temp.sort_values(len(temp), axis=1, ascending=False)
    #drop the row that has the values
    temp_sorted = temp_sorted.drop(temp_sorted.tail(1).index)
    #keep the 'max_number_of_movies' most rated movies
    return_df = temp_sorted.iloc[:, :max_number_of_movies]
    return return_df.columns


def clusterUsers(movie_rating_df_pu, movie_avg_rating_df):
    '''This function clusters the users together so we can fill any NaN values in user_rating
    note: I did not use Standard Scaler because all dimensions represent movies and the values are user ratings.
    Thus, the distance in every dimension is in the same metric. No need to scale or normalize.'''
    # Before clustering we need to fill NaN values. I chose to do this based on the avg rating per unique combination of movie genre
    movie_rating_df_pu_noNaN = fillNanWithAvgGenreRating(movie_rating_df_pu, movie_avg_rating_df)
    # X below pivoted df still has Null Values, which we will fill after the clustering
    X = movie_rating_df_pu.pivot(index='userId', columns='movieId', values='rating')
    # X_noNaN, pivoted, has no NaN values (filled based on genre), which we will use to create the clusters. 
    X_noNaN = movie_rating_df_pu_noNaN.pivot(index='userId', columns='movieId', values='rating')
    #get the 1000 most rated movies from the dataframe that has NaN values
    best_movie_columns = getMostRatedMovieColumns(X,1000)
    X_best_noNaN = X_noNaN[best_movie_columns]
    #run Kmeans and get final predictions
    predictions = predictWithKmeans(6, X_best_noNaN)   
    #combine predictions with initial dataframe X that contains all user-movie-ratings
    combineWithCluster(X, predictions)
    return X

def predictWithKmeans(clusters, matrix):
    '''This functions clusters any data in the matrix using KMEANS with k=clusters.'''
    #random stat 8 seems to do the best split
    return KMeans(n_clusters = clusters, algorithm='full', random_state=2).fit_predict(matrix)


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