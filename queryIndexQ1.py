'''
After Elasticsearch 5.0 the default similarity algorithm is BM25
'''
from elasticsearch import Elasticsearch
 
es = Elasticsearch()

print('type "exit" if you want to exit the search')
user_input = input("What movie do you want? (by title): \n")
while(user_input != 'exit'):

    
    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(user_input)}}}, size = 1000)
    hits = res['hits']['total']['value']
    print("Got {} Hits:".format(hits))

    # I set the maximum number of results to 1000. The default was 10. So we need to take that into account
    try :
        for i in range(hits):
            print(i+1,') ',res['hits']['hits'][i]['_source']['title'])
    except:
        for i in range(1000):
            print(i+1,') ',res['hits']['hits'][i]['_source']['title'])

    print('type "exit" if you want to exit the search')
    user_input = input("What movie do you want? (by title): \n")

    

