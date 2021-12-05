import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from nltk.corpus import wordnet as wn
try:
    from pymagnitude import *
except:
    from pymagnitude import *
 
from scipy.spatial import distance
from hashlib import blake2b
import random
from elasticsearch import Elasticsearch
from nltk.corpus import wordnet   #Import wordnet from the NLTK
from nltk.stem import WordNetLemmatizer
    


class DBSCAN(ClusterMixin, BaseEstimator):
    
    def __init__(self, eps=0.5, *, metric='euclidean',
                 metric_params=None, 
                allwords=[],cluster_name_index="no_cluster_name", word_embedding_path="/home/paul//mots.magnitude"):
        self.eps = eps
        self.metric = metric
        self.metric_params = metric_params
        self.clusters = []
        self.vectors = Magnitude(word_embedding_path)
        self.allwords =allwords
        self.es = Elasticsearch()
        self.cluster_name_index=cluster_name_index
        self.wordnet_lemmatizer = WordNetLemmatizer()


    
    
    def gethash(self,word):
        h = blake2b(digest_size=35)
        h.update(str(word).encode('utf-8'))
        return h.hexdigest()   
    
    def add_cluster(self,word,is_pronom,word_typ):
        point ={
            "word":word,
            "in_cluster":False,
            "cluster_id":None,
            }
        
         
        if len(self.clusters)==0 or is_pronom==True or word_typ=="NUM":
            wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
            idhas = self.gethash(wordhas)
            point["cluster_id"] = idhas
            self.allwords.append(point)
            cluster = {
             'id': idhas,
             'centroid': self.vectors.query(word),
              'words': [word],
                }
            self.clusters.append(cluster)
            # self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
            
        else:
            clustermin_index = None
            vector_word = self.vectors.query(word)
            val_min = distance.cosine(self.clusters[0]["centroid"],vector_word)
            for index, cluster in enumerate(self.clusters):
                dis=distance.cosine(cluster["centroid"],vector_word)
                if dis < self.eps and val_min >= dis:
                    clustermin_index = index
                    val_min = dis

            
            if clustermin_index == None:
                wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = self.gethash(wordhas)
                point["cluster_id"] = idhas
                self.allwords.append(point)
                cluster = {
                'id': idhas,
                'centroid': vector_word,
                'words': [word],
                    }
                self.clusters.append(cluster)
               # self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
                
            else:
                if word_typ=="ADJ" or word_typ=="NOUN" or word_typ=="VERB" or word_typ=="ADV":
                    if self.get_ontonym(self.clusters[clustermin_index]["words"][0],word)==True:
                        wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
                        idhas = self.gethash(wordhas)
                        point["cluster_id"] = idhas
                        self.allwords.append(point)
                        cluster = {
                         'id': idhas,
                         'centroid': self.vectors.query(word),
                          'words': [word],
                            }
                        self.clusters.append(cluster)
                    else:
                        self.clusters[clustermin_index]["words"].append(word)
                else:
                    self.clusters[clustermin_index]["words"].append(word)
                
               # self.es.index(index=self.cluster_name_index, id=self.clusters[clustermin_index]["id"], body=self.clusters[clustermin_index])
    def get_ontonym(self,word1,word2):
        ant = list()
        wordlema = self.wordnet_lemmatizer.lemmatize(word1, pos="v")
        wordlema2 = self.wordnet_lemmatizer.lemmatize(word2, pos="v")
        for synset in wordnet.synsets(wordlema):
            for lemma in synset.lemmas():
                if lemma.antonyms(): 
                #When antonyms are available, add them into the list
                    ant.append(lemma.antonyms()[0].name())
        return wordlema2 in ant

    
   
    
    
    