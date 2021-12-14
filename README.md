# Documentation
## _The Last Markdown Editor, Ever_


Le projet contient 4 principauxfichiers:

- dbscan.py 
- model.py
- processing.py
- tools.py

## dbscan.py
#### méthodes
c'est dans cette classe que se trouve les scripts de construction des clusters. On y trouve la classe DBSCAN avec pour principale méthode:
- add_word
cette méthode permet d'ajouter un mots aux clusters selon la méthode incrémental dbscan de base pour un minexemple=2
- add_cluster: 
cette méthode permet d'ajouter un mot au clusters en utilisant le sécond algorithme.

#### propriétés
- eps: seil pour la construction des clusters (1-cosinus(alpha)
- min_samples : nombre minimum d'élements que doit contenir un clusters
- clusters: clusters ainsi formés
- allwords: tous les mots du vocabulaire insérés dans les clusters
- cluster_name_index le nom de l'index où est stoké les clusters dans elasticsearch
Les commentaire des autres algorithme se trouve dans le code.

## model.py
dans se fichier se trouve la class Vectorisation qui est chargé d'effectuer l'ensemble des prétraitement sur données et indexé ceux-ci.
#### Propriétés:
- clusters: les clustés formés par la classe dbscan
- documents: l'ensemble des documents sous forme d'une liste
- eps: le seil utilisé pour la construction des clusters
- index_name : le nom de l'indexe où seront stoké les documents dans l'élasticsearch
#### Méthodes:
A ce niveau il y'a plusieurs petites méthoses qui sont combiner pour indexer les documents j'ai laissé un petit commentaire dans le corde:
- fit cette méthodes prend en paramètre une liste de documents ainsi qu'une liste d'ids et appelle les sous méthodes pour effectuer tout le precessus d'indexation puis elle retourne l'instance du model.
- get_res_query(query_text): à partir d'une requête au format texte retourne les 50 résultats correspondant
- buil_query(text): permet d'indexer une requête
- get_cluster(word): retourne l'indice du cluster dans lequel se trouve un mots ainsi que ça proximité avec le centradoide
- get_tokens(text): effectue l'ensemble des prétraitemens sur le texte brut et retouner les tokens
- create_index(): cree l'index elasticsearch 
- create_bm25_index(): creer l'index associé à bm25
- model_combine(self,res_my,resbm24): combine le resultat du model et celui de bm25
## preprocessing.py
Cette classe effectue tout le travail en appelant les deux classes précedentes, elle est sansé être la façade de tout le système et caché toutes les implémentations faites en amont.
#### Proprités:
- timeout: le temps au bout duquel une connexion elastic search sera considéré comme échouée
- documents la liste de documents
- eps: le seuil utilisé pour le construction des clusters
- index_name : le nom de l'index où seront stocké les documents dans elasticsearch
- magnitue_path: le chemin vers le word embedding
- db: une instance de la classe DBSCAN précedente

#### Méthodes
en cours...




## Installation 
pip install -r requirements.txt
nltk.download('wordnet')
nltk.download('stopwords')


## Requirement
### elasticsearch 
You must have Elasticsearch installed on your machine the documents are indexed directly in elasticsearch to speed up the search 
### vector embeddings fasttest
We use word embedding fasttext to semantically group the words in the collection into a single cluster so you need to specify the location of the Elasticsearch file on disk   

you can download it fasttext word embedding **[`here`](https://fasttext.cc/docs/en/english-vectors.html)**:

## Test
To be able to test the api you have to start Elasticsearch and place the word embedding file in the folder with the name words.vectors because the test script here directly looks for this file.

the file launch.py contains the script to test the model. The test consists in indexing a part of data of train-v2 Squard and launching a small server which listens a request on the port 5002 

Use the post method with a json {request: "text_request"} as body to get the results corresponding to "text_request

you can then customize the script in the launch.py file to fit your needs



