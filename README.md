# elasticsearch-movies

A search engine for movies from the [MovieLens](https://movielens.org/) dataset improved by user profiles created by clustering and Machine Learning techniques.  
The goal was to familiarize myself with Elasticsearch and play around with some ML models that could improve the ranking.

The code is broken down into 5 python files and a Jupyter Notebook.  
Each improves on the last one by adding an additinal complexity to the Ranking algorithm for the movies.  

The final ranking score when querying for a movie is calculated as follows :  

[BM25 score](https://en.wikipedia.org/wiki/Okapi_BM25) + User's movie rating + User's cluster movie rating + Random Forest Classifier prediction 


![example](./example.gif)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to run the project and replicate results.

### Prerequisites and Installation

A step by step series of examples that tell you how to get a development env running

You can run the following on the command prompt to install any requirements :

```
pip install -r requirements.txt  
```
If the above fails it is recommended that you download [Anaconda]() which comes packaged with most of the libraries
we will be needing and then you only need to open Anaconda prompt and 
```
conda install elasticsearch
```
If you dont want Anaconda run the following in cmd
``` 
pip install -U Cython
pip install -U scipy
pip install -U matplotlib
pip install -U pandas
pip install -U scikit-learn
pip install elasticsearch
```

### Deployment

Before running any files an Elasticsearch server need to be running.
So after installing [Elasticsearch](https://www.elastic.co/downloads/elasticsearch), go to the installation folder
and run :
```
./bin/elasticsearch
```
If you want a [Kibana](https://www.elastic.co/downloads/kibana) server up and running to send API requests to Elasticsearch you can instal Kibana and once again run :
```
./bin/kibana
```

Also if you are planning on running queryIndexQ4 and q4PreProcessing_notebook make sure to create the following folders in the file structure :  

./data/training/  
./data/predictions/  
./data/prediction_result/  


## Details

* **createIndexQ1** 
  * creates an Elasticsearch Index from movies.csv and ratings.csv files. You only need to run this.

* **queryIndexQ1**
  * Only takes into account BM25 score from Elasticsearch and returns a ranking based on user input.

* **queryIndexQ2**
  * Also takes into account which user is searching, thus the final ranking's score is BM25 score + user's movie rating.

* queryIndexQ3 
  * clusters users together using K-means based on their rating on common movies. Also takes into accoun the genre of the movie. Thus, the ranking score here is BM25 score + user's movie rating + movie rating's from cluster

* **queryIndexQ4**
  * Also takes into account an [RFC](https://en.wikipedia.org/wiki/Random_forest) model prediction for each user/movie combination. An RFC is trained for each user with datasets created by the tf-idf vectors of the title and a one-hot-encoding from the movie's genre.
  * The models are created on the Jupyter Notebook and imported here with pickle.
  * **(note)**: This isn't worth it and does not improve the final ranking. SVM's would be better plus training on movie tf-idf scores from the title is not good. I only did this step to work more with Elasticsearch's API and its built-in term_frequency and document_frequency numbers.

* **q4PreProcessing_notebook**
  * Does all the dataset creattion, model training and additional caclulations needed to create the final models stored as pickle files.

**SOS**   
To run queryIndexQ4 you must first run queryIndexQ3 and q4PreProcessing_notebook which create the initial user clustering and train a RandomForest model for each user respectively. Those are saved as pickle files in ./data/ and are used in queryIndexQ4 for the final ranking.

## Built With

* [Elasticsearch](https://www.elastic.co/) - The open-souce search engine used
* [Kibana](https://www.elastic.co/kibana) - A GUI for Elasticsearch so I can test some API requests
* [Python and ML libraries](https://www.python.org/) - Jupyter Notebooks, Pandas, Scikit-Learn, Pickle etc.. 
