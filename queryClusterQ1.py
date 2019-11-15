from elasticsearch import Elasticsearch
 
es = Elasticsearch()

user_input = input("What movie do you want? (by title): \n")

res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(user_input)}}})
hits = res['hits']['total']['value']
print("Got {} Hits:".format(hits))

for i in range(hits):
    print(i+1,') ',res['hits']['hits'][i]['_source']['title'])