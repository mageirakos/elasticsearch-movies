'''
Requirements :
    Elasticsearch -> pip install elasticsearch 


(SHIFT+RightClick to start powershell on folder)
To start elasticsearch server :
1) Run ./bin.elasticsearch.bat 
2) http://localhost:9200

To start Kibana server:
1) Run ./bin/kibana.bat
2) http://localhost:5601

> 
'''

# You only need to run this once to create the index


from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()

with open('./data/movies.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='movies', doc_type='my-type')

