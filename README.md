# Clustering Information Retrieval
This repository is dedicated to the implementation of a Semantic Information Retrieval System which
group word into cluster using word embeddings.
Documents and queries are then projected in the vector space form by clusters.
At the end, results obtained with this approach are combined with bm25 ones.

A Flask API (launch.py) is developed to illustrate the use this approach on the SQuAD dataset.
It can be use to retrieve paragraphs relevant to a query.

## Package Installation 
in a Python 3 Environment
```
pip install -r requirements.txt
nltk.download('wordnet')
nltk.download('stopwords')

```


## System Requirements
- Elasticsearch running on port 9200 with password security
- Fasttext embeddings (PyMagnitude) at location "/home/user/embeddings.magnitude"

## Running the script
```
python launch.py

```
this will expose the API on port 5002 of localhost


